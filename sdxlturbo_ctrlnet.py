#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch
import time
from diffusers import AutoencoderTiny
from sfast.compilers.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)
from diffusers.utils import load_image
import xformers
import triton
import lunar_tools as lt
from PIL import Image
import numpy as np
from diffusers.utils.torch_utils import randn_tensor
import random as rn
import numpy as np
import xformers
import triton
import cv2
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from prompt_blender import PromptBlender
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

#%% PARAMETERS

ctrlnet_type = "diffusers/controlnet-canny-sdxl-1.0"
# ctrlnet_type = "diffusers/controlnet-depth-sdxl-1.0"
use_maxperf = False
shape_cam=(600,800)
size_ctrl_img = (512, 512) 
num_inference_steps = 2
controlnet_conditioning_scale = 0.45
stitch_cam = False

# %% INITS
cam_man = lt.WebCam(cam_id=0, shape_hw=shape_cam)
cam_man.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

if "depth" in ctrlnet_type:
    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

# initialize the models and pipeline
controlnet = ControlNetModel.from_pretrained(
    ctrlnet_type,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
pipe.vae = pipe.vae.cuda()
pipe.set_progress_bar_config(disable=True)

if use_maxperf:
    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    config.enable_jit = True
    config.enable_jit_freeze = True
    config.trace_scheduler = True
    config.enable_cnn_optimization = True
    config.preserve_parameters = False
    config.prefer_lowp_gemm = True
    pipe = compile(pipe, config)

#%%
    
def stitch_images(img1, img2):
    # Determine the size for the new image
    width1, height1 = img1.size
    width2, height2 = img2.size
    new_width = width1 + width2
    new_height = max(height1, height2)

    # Create a new image with appropriate size
    new_img = Image.new('RGB', (new_width, new_height))

    # Paste the images side by side
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (width1, 0))

    return new_img


def process_cam_image(ctrl_image, ctrlnet_type):
    ctrl_image = np.array(ctrl_image)
    
    if ctrlnet_type == "diffusers/controlnet-canny-sdxl-1.0":
        low_threshold = 100
        high_threshold = 200
        ctrl_image = cv2.Canny(ctrl_image, low_threshold, high_threshold)
        ctrl_image = ctrl_image[:, :, None]
        ctrl_image = np.concatenate([ctrl_image, ctrl_image, ctrl_image], axis=2)
        ctrl_image = Image.fromarray(ctrl_image)
    else:
        image = feature_extractor(images=ctrl_image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = depth_estimator(image).predicted_depth
    
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=size_ctrl_img,
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)
    
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        ctrl_image = image
        
    return ctrl_image

def pad_image_to_width(image, width_renderer):
    """
    Pads the given PIL image on the left and right with zeros to reach a specified width.

    Args:
    image (PIL.Image): The image to be padded.
    width_renderer (int): The final width of the image after padding.

    Returns:
    PIL.Image: The padded image.
    """
    current_width, _ = image.size
    total_padding = width_renderer - current_width
    if total_padding < 0:
        raise ValueError("width_renderer must be greater than the current image width.")

    # Divide the padding equally on both sides
    pad_width = total_padding // 2

    img_array = np.array(image)

    padded_array = np.pad(img_array, ((0, 0), (pad_width, pad_width), (0, 0)), mode='constant')

    # Convert the padded array back to a PIL image
    return Image.fromarray(padded_array)


def center_crop_and_resize(img, size=(512, 512)):
    """
    Center crop an image to the specified size and resize it.

    Args:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the cropped and resized image.
        size (tuple): Target size in the format (width, height). Default is (512, 512).
    """
    try:
        # Get the original dimensions of the image
        width, height = img.size

        # Calculate the coordinates for the center crop
        left = (width - size[0]) / 2
        top = (height - size[1]) / 2
        right = (width + size[0]) / 2
        bottom = (height + size[1]) / 2

        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))

        # Resize the cropped image to the specified size
        cropped_img = cropped_img.resize(size)

        return cropped_img
    except Exception as e:
        print(f"An error occurred: {e}")



#%%
# Example usage
blender = PromptBlender(pipe)

# base = 'skeleton person head skull terrifying'
prompt = 'very bizarre and grotesque zombie monster'
prompt = 'very bizarre alien with spaceship background'
prompt = 'a funny weird frog'
prompt = 'metal steampunk gardener'
prompt = 'a strange city'
prompt = 'a strange football match'
prompt = 'a very violent boxing match'
prompt = 'ballet dancing'
prompt = 'a beautiful persian belly dancers'
prompt = 'a medieval scene with people dressed in strange clothes'
prompt = 'a bird with two hands'
# base = 'a beautiful redhaired mermaid'
# base = 'terror pig party'
# base = 'dirty and slimy bug monster'
# base = 'a telepathic cyborg steampunk'
# base = 'a man wearing very expensive and elegant attire'
# base = 'a very beautiful young queen in france in versaille'
# base = 'a gangster party'


sz = (size_ctrl_img[0]*2, size_ctrl_img[1]*2)
width_renderer = width=2*sz[1]

renderer = lt.Renderer(width=width_renderer, height=sz[0])
    
latents = torch.randn((1,4,64//1,64)).half().cuda()

prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt)

speech_detector = lt.Speech2Text()

akai_lpd8 = lt.MidiInput("akai_lpd8")

while True:
    torch.manual_seed(1)
    
    num_inference_steps = int(akai_lpd8.get("E1", val_min=1, val_max=5, val_default=1))
    
    controlnet_conditioning_scale = akai_lpd8.get("E0")

    if akai_lpd8.get("A0", button_mode="is_held"):
        if not speech_detector.audio_recorder.is_recording:
            speech_detector.start_recording()
    elif not akai_lpd8.get("A0", button_mode="is_held"):
        if speech_detector.audio_recorder.is_recording:
            prompt = speech_detector.stop_recording()
            print(f"New prompt: {prompt}")
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt)
            stop_recording = False

    cam_img = cam_man.get_img()
    cam_img = np.flip(cam_img, axis=1)
    cam_img = Image.fromarray(np.uint8(cam_img))
    cam_img = center_crop_and_resize(cam_img)
    
    ctrl_img = process_cam_image(cam_img, ctrlnet_type)

    image = pipe(image=ctrl_img, latents=latents, controlnet_conditioning_scale=controlnet_conditioning_scale, guidance_scale=0.0, num_inference_steps=num_inference_steps, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]
    
    stitch_cam = akai_lpd8.get("D0", button_mode="toggle")
    show_ctrlnet_img = akai_lpd8.get("D1", button_mode="toggle")
    
    if stitch_cam:
        if show_ctrlnet_img:
            image = stitch_images(ctrl_img, image)
        else:
            image = stitch_images(cam_img, image)
            
    else:
        image = pad_image_to_width(image, image.size[0]*2)
        
    renderer.render(image)
        
"""
WISHLIST
- threaded controlnet computation
- other controlnets
- mess with more prompts

"""