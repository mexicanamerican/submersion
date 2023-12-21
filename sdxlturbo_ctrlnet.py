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
stitch_cam = True

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

# Example usage
blender = PromptBlender(pipe)

from nltk.corpus import wordnet as wn
nouns = list(wn.all_synsets('n'))
nouns = [nouns[i].lemma_names()[0] for i in range(len(nouns))]

# base = 'skeleton person head skull terrifying'
base = 'very bizarre and grotesque zombie monster'
base = 'very bizarre alien with spaceship background'
base = 'a funny weird frog'
base = 'metal steampunk gardener'
base = 'a strange city'
base = 'a beautiful redhaired mermaid'
base = 'a gangster party'
base = 'terror pig party'
base = 'dirty and slimy bug monster'
base = 'a telepathic cyborg steampunk'
base = 'a man wearing very expensive and elegant attire'

tp = 150
prompts = []
for i in range(tp):
    prompts.append(f'A painting of {base} who looks like {nouns[np.random.randint(len(nouns))]}. 4K Ultra HD. A lot of detail')

n_steps = 30
blended_prompts = blender.blend_sequence_prompts(prompts, n_steps)

# Image generation pipeline


sz = (size_ctrl_img[0]*2, size_ctrl_img[1]*2)
if stitch_cam:
    width_renderer = width=2*sz[1]
else:
    width_renderer = width=1*sz[1]
    
renderer = lt.Renderer(width=width_renderer, height=sz[0])
    
latents = torch.randn((1,4,64//1,64)).half().cuda()

# Iterate over blended prompts
while True:
    print('restarting...')
    
    for i in range(len(blended_prompts) - 1):
        torch.manual_seed(1)
        
        fract = float(i) / (len(blended_prompts) - 1)
        blended = blender.blend_prompts(blended_prompts[i], blended_prompts[i+1], fract)
    
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blended
        
        cam_img = cam_man.get_img()
        cam_img = np.flip(cam_img, axis=1)
        cam_img = Image.fromarray(np.uint8(cam_img))
        cam_img = center_crop_and_resize(cam_img)
        
        ctrl_img = process_cam_image(cam_img, ctrlnet_type)
        
        image = pipe(image=ctrl_img, latents=latents, controlnet_conditioning_scale=controlnet_conditioning_scale, guidance_scale=0.0, num_inference_steps=num_inference_steps, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]
        
        # Render the image
        if stitch_images:
            image = stitch_images(cam_img, image)
        renderer.render(image)
        
"""
WISHLIST
- threaded controlnet computation
- other controlnets
- mess with more prompts

"""