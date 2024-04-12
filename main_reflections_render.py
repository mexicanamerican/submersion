#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch
import time
from diffusers import AutoencoderTiny
import xformers
import triton
import lunar_tools as lt
from PIL import Image
import numpy as np
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
import cv2
import sys
sys.path.append("../psychoactive_surface")
from prompt_blender import PromptBlender
from typing import List, Tuple
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters.kernels import get_binary_kernel2d
from img_utils import pad_image_to_width, pad_image_to_width, blend_images, process_cam_img, stitch_images, weighted_average_images
from datetime import datetime
from human_seg import HumanSeg
torch.set_grad_enabled(False)
from img_utils import center_crop_and_resize
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


from diffusers import AutoPipelineForInpainting


def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    computed: List[int] = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]

def median_blur(input: torch.Tensor,
                kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Blurs an image using the median filter.

    Args:
        input (torch.Tensor): the input image with shape :math:`(B,C,H,W)`.
        kernel_size (Tuple[int, int]): the blurring kernel size.

    Returns:
        torch.Tensor: the blurred input tensor with shape :math:`(B,C,H,W)`.

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> output = median_blur(input, (3, 3))
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    padding: Tuple[int, int] = _compute_zero_padding(kernel_size)

    # prepare kernel
    kernel: torch.Tensor = get_binary_kernel2d(kernel_size).to(input)
    b, c, h, w = input.shape

    # map the local window to single vector
    features: torch.Tensor = F.conv2d(
        input.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1)
    features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

    # compute the median along the feature axis
    median: torch.Tensor = torch.median(features, dim=2)[0]

    return median

class MedianBlur(nn.Module):
    r"""Blurs an image using the median filter.

    Args:
        kernel_size (Tuple[int, int]): the blurring kernel size.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = MedianBlur((3, 3))
        >>> output = blur(input)
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """

    def __init__(self, kernel_size: Tuple[int, int]) -> None:
        super(MedianBlur, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return median_blur(input, self.kernel_size)
#%% PARAMETERS
compile_with_sfast = True
shape_cam= (1080, 1920)
size_diff_img = (512, 512) 

# %% INITS
cam = lt.WebCam(cam_id=0, shape_hw=shape_cam)
cam.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)

blur = MedianBlur((3, 3)) 

pipe = AutoPipelineForInpainting.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    
pipe = pipe.to("cuda")
# pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
# pipe.vae = pipe.vae.cuda()
pipe.set_progress_bar_config(disable=True)

if compile_with_sfast:
    from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)
    pipe.enable_xformers_memory_efficient_attention()
    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    pipe = compile(pipe, config)

#%%


#%%
# Example usage
blender = PromptBlender(pipe)
generator = torch.Generator(device="cuda")

human_seg = HumanSeg()

prompt = "photo of a happy person"
negative_prompt = 'blurry, tiled, wrong, bad art, pixels, amateur drawing, haze'
    
image_diffusion = Image.fromarray(np.zeros((size_diff_img[1], size_diff_img[0], 3), dtype=np.uint8))

prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)

renderer = lt.Renderer(width=shape_cam[1], height=shape_cam[0])
speech_detector = lt.Speech2Text()
meta_input = lt.MetaInput()

fps = 0

currently_recording_vid = False
cam_img_first = None
blob_image = None

while True:
    torch.manual_seed(420)
    num_inference_steps = int(meta_input.get(akai_midimix="D0", akai_lpd8="H1", val_min=2, val_max=5, val_default=2))
    
    do_record_mic = meta_input.get(akai_midimix="A3", akai_lpd8="A0", button_mode="held_down")
    if do_record_mic:
        if not speech_detector.audio_recorder.is_recording:
            speech_detector.start_recording()
    elif not do_record_mic:
        if speech_detector.audio_recorder.is_recording:
            try:
                prompt = speech_detector.stop_recording()
            except Exception as e:
                print(f"FAIL {e}")
            print(f"New prompt: {prompt}")  
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)
            stop_recording = False
            
    do_record_vid = meta_input.get(akai_lpd8="A1", button_mode="toggle")
    if do_record_vid and not currently_recording_vid:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%y%m%d_%H%M")
        fp_out = f"{formatted_datetime}.mp4"
        movie = lt.MovieSaver(fp_out, fps=fps)
        currently_recording_vid = True
    
    elif currently_recording_vid and not do_record_vid:
        movie.finalize()
        currently_recording_vid = False
    
    fract_mixing = 0

    cam_img = cam.get_img()
    cam_img = np.flip(cam_img, axis=1)
    cam_img = np.uint8(cam_img)
    
    # Crop an image out that is the size of what we put into diffusion
    cam_crop_x = int(meta_input.get(akai_midimix="C1", val_min=0, val_max=shape_cam[1]-size_diff_img[1]))
    cam_crop_y = int(meta_input.get(akai_midimix="C2", val_min=0, val_max=shape_cam[0]-size_diff_img[0]))
    cam_img_cropped = cam_img[cam_crop_y:cam_crop_y+size_diff_img[0], cam_crop_x:cam_crop_x+size_diff_img[1], :]
    cam_img = Image.fromarray(cam_img)
    cam_img_cropped = Image.fromarray(cam_img_cropped)
    
    # Within this image we make a square mask
    mask = np.zeros(size_diff_img, dtype=np.uint8)
    size_inlet = int(meta_input.get(akai_midimix="C0", val_min=50, val_max=450))
    
    # Calculate the center of the mask
    center_y, center_x = [dim // 2 for dim in mask.shape]
    # Calculate the top left corner of the square
    y_inlet = center_y - size_inlet // 2
    x_inlet = center_x - size_inlet // 2
    # Create the square in the center of the mask
    mask[y_inlet:y_inlet+size_inlet, x_inlet:x_inlet+size_inlet] = 255
    mask = Image.fromarray(mask)

    # human_mask = human_seg.get_mask(np.asarray(cam_img))
    # human_mask = human_mask.astype(np.float32)
    
    # mask = human_mask


    guidance_scale = 0.0
    t0 = time.time()
    
    generator = torch.Generator(device="cuda").manual_seed(0)    
    image_diffusion = pipe(image=cam_img_cropped, generator=generator, mask_image=mask, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]
    
    # Re-insert image_diffusion into cam_img at the cropped position
    cam_img.paste(image_diffusion, (cam_crop_x, cam_crop_y))
    
    
    dt = time.time() - t0
    fps = 1/dt
    # print(f"\rCurrent fps: {fps:.1f}", end="")
    
    # stitch_cam = meta_input.get(akai_lpd8="D0", button_mode="toggle")
    
    show_cam = meta_input.get(akai_midimix="H3", akai_lpd8="D1", button_mode="toggle")
    # use_mask = True
    # 

    
    # if use_mask:
        
    #     # fract_ramp = meta_input.get(akai_midimix="H5", val_default=0, val_min=0, val_max=1.0)
    #     # fract_ramp = 1 - np.clip(different_pixels_count/1000, 0, 1)
        
    #     image_diffusion_ramped = np.asarray(image_diffusion).astype(np.float32)
    #     image_diffusion_ramped *= fract_ramp
    #     image_diffusion_ramped += (1-fract_ramp) * np.asarray(cam_img)
                
    #     image = np.zeros_like(image_diffusion)
    #     image = mask_last[..., np.newaxis] * image_diffusion_ramped + (1 - mask_last[..., np.newaxis]) * cam_img_last
    #     image_diffusion = Image.fromarray(image.astype(np.uint8))
    
    
    if show_cam:
        image_show = pad_image_to_width(cam_img_orig, image_diffusion.size[0]*2)
    else:
        image_show = cam_img #pad_image_to_width(cam_img, image_diffusion.size[0]*2)
        
    renderer.render(image_show)
    
    if do_record_vid and currently_recording_vid:
        movie.write_frame(np.asarray(image_show))
    

