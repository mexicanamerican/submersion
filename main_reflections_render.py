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
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import random

model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
yolo = YOLO(model_path)
#%% PARAMETERS
compile_with_sfast = False
shape_cam= (1080, 1920)
size_diff_img = (512, 512)
size_render = shape_cam #(1024, 1024) 
padding = 70
human_mask_boundary_relative = 0.1
guidance_scale = 0.0
# %% INITS
cam = lt.WebCam(cam_id=0, shape_hw=shape_cam)
cam.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)

pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", local_files_only=True)
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

# %% aux
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


def crop_face_square(cam_img, padding=30):
    try:
        # run face detection
        face_results = yolo(cam_img, verbose=False)
    #    If no face present, no need to do anything
        if len(face_results)==0:
            renderer.render(cam_img)
            return None
        
        # Initialize variables to store the maximum area and corresponding index
        max_area = 0
        index_of_largest_face = -1
        cropping_coordinates = None
        
        # Loop through all detected faces
        for i, result in enumerate(face_results):
            # Retrieve the bounding box coordinates in the format (x1, y1, x2, y2)
            x1, y1, x2, y2 = result.boxes.xyxy[0].cpu().numpy()
            # Calculate the area of the bounding box
            area = (x2 - x1) * (y2 - y1)
            
            # Check if this area is the largest we've seen so far
            if area > max_area:
                max_area = area
                index_of_largest_face = i
                # Update the cropping coordinates to the largest face found
                
                
        # Adjust to square
        
        # Calculate width and height of the detected face
        width = x2 - x1
        height = y2 - y1
        
        # Determine the amount to extend to make the bounding box square
        if width > height:
            difference = (width - height) // 2
            y1 = max(0, y1 - difference)  # Extend upwards
            y2 = min(cam_img.shape[0], y2 + difference)  # Extend downwards
        elif height > width:
            difference = (height - width) // 2
            x1 = max(0, x1 - difference)  # Extend to the left
            x2 = min(cam_img.shape[1], x2 + difference)  # Extend to the right
        
        # Adjust dimensions to make the bounding box square
        if (x2 - x1) != (y2 - y1):  # Check if not square due to clamping at image borders
            new_size = min(x2 - x1, y2 - y1)
            x2 = x1 + new_size
            y2 = y1 + new_size
        
        # Calculate new padding to center the padding around the box
        padding_x = (padding // 2, padding // 2)
        padding_y = (padding // 2, padding // 2)
        
        # Adjust padding if it causes the box to go out of image bounds
        if x1 - padding_x[0] < 0:
            padding_x = (x1, padding - x1)
        if x2 + padding_x[1] > cam_img.shape[1]:
            padding_x = (padding - (cam_img.shape[1] - x2), cam_img.shape[1] - x2)
        
        if y1 - padding_y[0] < 0:
            padding_y = (y1, padding - y1)
        if y2 + padding_y[1] > cam_img.shape[0]:
            padding_y = (padding - (cam_img.shape[0] - y2), cam_img.shape[0] - y2)
        
        # Apply adjusted padding
        x1 = max(0, x1 - padding_x[0])
        x2 = min(cam_img.shape[1], x2 + padding_x[1])
        y1 = max(0, y1 - padding_y[0])
        y2 = min(cam_img.shape[0], y2 + padding_y[1])
        
        cropping_coordinates = (int(x1), int(y1), int(x2), int(y2))
    
        return cropping_coordinates
    
    except Exception as e:
        return None
    

#%%
blender = PromptBlender(pipe)
human_seg = HumanSeg()
negative_prompt = 'blurry, tiled, wrong, bad art, pixels, amateur drawing, haze'
    
image_diffusion = Image.fromarray(np.zeros((size_diff_img[1], size_diff_img[0], 3), dtype=np.uint8))



latents = blender.get_latents()

renderer = lt.Renderer(width=size_render[1], height=size_render[0])
speech_detector = lt.Speech2Text()
meta_input = lt.MetaInput()

#%%


list_nationalities = ["nigerian", "moroccan", "turkish", "persian", "spanish", "american", "japanese", "chinese", "mongolian", "russian", "Brazilian", "Australian", "Egyptian", "Finnish", "Canadian", "Argentinian", "South Korean", "Kenyan", "Ukrainian", "Norwegian"]
list_emotions = ["angry", "sad", "surprised", "happy", "smiling", "furious", "Ecstatic", "Melancholic", "Bewildered", "Content", "Nostalgic", "Anxious"]
list_ages = ["middle aged", "very old", "very young", "teenager", "Toddler", "Elderly", "Young adult", "In their thirties"]

nationality = random.choice(list_nationalities)
emotion = random.choice(list_emotions)
age = random.choice(list_ages)
prompt = f"photo of a {emotion} and {age} {nationality} person"



#%%
human_mask_boundary = int(human_mask_boundary_relative*min(size_diff_img))
is_transformation_active = False
p_start_transform = 0.1
t_transform_started = 0

prompt = "photo of a very old and very angry american person"
prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)

while True:
    
    # handle the number of diffusion strength
    if not is_transformation_active:
        if np.random.rand() < p_start_transform:
            is_transformation_active = True
        
    get_new_prompt = meta_input.get(akai_midimix='B3', akai_lpd8='A0', button_mode='pressed_once')
    if get_new_prompt:
        nationality = random.choice(list_nationalities)
        emotion = random.choice(list_emotions)
        age = random.choice(list_ages)  
        prompt = f"photo of a {emotion} and {age} {nationality} person"
        print(f"new prompt: {prompt}")
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)
    
    t0 = time.time()
    torch.manual_seed(420)
    num_inference_steps = int(meta_input.get(akai_midimix="B2", akai_lpd8="H1", val_min=3, val_max=30, val_default=30))

    cam_img = cam.get_img()
    cam_img = np.flip(cam_img, axis=1)
    cam_img = np.uint8(cam_img)
    
    cropping_coordinates = crop_face_square(cam_img, padding=150)
    if cropping_coordinates is None:
        renderer.render(cam_img)
        time.sleep(0.05)
        # continue
    cam_img_cropped = Image.fromarray(cam_img).crop(cropping_coordinates)
    size_cam_img_cropped_orig = cam_img_cropped.size
    cam_img_cropped = cam_img_cropped.resize(size_diff_img)
    
    # only apply human mask in cam_img_cropped  
    human_mask = human_seg.get_mask(np.asarray(cam_img_cropped))

    strength = 1/num_inference_steps + 0.00001
    image_diffusion = pipe(image=cam_img_cropped, 
                    latents=latents, num_inference_steps=num_inference_steps, strength=strength, 
                    guidance_scale=guidance_scale, prompt_embeds=prompt_embeds, 
                    negative_prompt_embeds=negative_prompt_embeds, 
                    pooled_prompt_embeds=pooled_prompt_embeds,          
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]
    dt = time.time() - t0
    fps = 1/dt
    
    # print(fps)

    show_cam = meta_input.get(akai_midimix="H3", akai_lpd8="D1", button_mode="toggle")
    show_segmask = meta_input.get(akai_midimix="H4", akai_lpd8="D1", button_mode="toggle")
    
    if show_cam:
        image_show = cam_img
    elif show_segmask:
        image_show = np.asarray(cam_img).copy()
        # image_show[:,:,0] = np.array(human_mask) * 255
    else:
        # just show image_diffusion_mixed for now
        # Re-insert image_diffusion into cam_img at the cropped position
        human_mask = human_mask.astype(np.float32)
        human_mask[-human_mask_boundary:,:] = 0
        human_mask[:,-human_mask_boundary:] = 0
        human_mask[:,:human_mask_boundary] = 0
        human_mask[:human_mask_boundary,:] = 0
        human_mask = cv2.GaussianBlur(human_mask, (55, 55), 0)
        if human_mask.max() > 0:
            human_mask = human_mask/human_mask.max()
        image_diffusion_mixed = blend_images(np.array(image_diffusion), cam_img_cropped, human_mask)
        
        image_diffusion_mixed_resized = image_diffusion_mixed.resize(size_cam_img_cropped_orig)
        cam_img_pil = Image.fromarray(cam_img)
        cam_img_pil.paste(image_diffusion_mixed_resized, cropping_coordinates[:2])
        image_show = cam_img_pil
        
    renderer.render(image_show)
    


