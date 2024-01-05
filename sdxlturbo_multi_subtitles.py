#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:13:24 2023

@author: lunar
"""

#%%
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch

from diffusers import AutoencoderTiny
from sfast.compilers.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)
from diffusers.utils import load_image
import lunar_tools as lt
from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prompt_blender import PromptBlender

#%%

def add_subtitle_to_frame(image, text, position, font_path='/usr/share/fonts/truetype/msttcorefonts/Arial.ttf', font_size=30, font_color=(255, 255, 255)):
    """
    Add a subtitle to an image.

    Args:
    image (np.array): The image to which the subtitle will be added.
    text (str): The subtitle text.
    position (tuple): The position (x, y) of the text on the image.
    font_path (str): Path to the font file.
    font_size (int): Size of the font.
    font_color (tuple): Color of the font in RGB.

    Returns:
    np.array: The image with the subtitle added.
    """
    # Convert the NumPy array to a PIL image
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)

    # Load a font
    font = ImageFont.truetype(font_path, font_size)

    # Add text to the image
    draw.text(position, text, font=font, fill=font_color)

    # Convert back to NumPy array
    return np.array(pil_img)


#%%


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

use_maxperf = False
use_image_mode = False

if use_image_mode:
    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
else:
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")

pipe.to("cuda")
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
torch.manual_seed(1)

# Example usage
blender = PromptBlender(pipe)
    
prompts = []
prompts.append('analogue film burn, texture')
prompts.append('analogue film, burned film, strong black, white spots of light')
prompts.append('analogue film chemical reaction distortion burned')

n_steps = 100
blended_prompts = blender.blend_sequence_prompts(prompts, n_steps)
noise_level = 0
l_size = [128,128]
latents = torch.randn((1,4,l_size[0]//1,l_size[1])).half().cuda() # 64 is the fastest
img_res = np.asarray(l_size)*8

#%%
# Image generation pipeline
sz = (512*1, 512*1)
#sz = (1920, 1080)
renderer = lt.Renderer(width=sz[1], height=sz[0])

image_res = 

# Iterate over blended prompts

n_frames = 100
frame = 0
#ms = lt.MovieSaver("my_movie.mp4", fps=10)

subtitle_text = 'TESTING SUBTITLES'

while True:
    print('restarting...')
    
    for i in range(len(blended_prompts) - 1):
        fract = float(i) / (len(blended_prompts) - 1)
        blended = blender.blend_prompts(blended_prompts[i], blended_prompts[i+1], fract)
    
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blended
            
        image = pipe(guidance_scale=0.0, num_inference_steps=1, latents=latents, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]
        
        # Render the image
        image = np.asanyarray(image)
        image = np.uint8(image)
        image = add_subtitle_to_frame(np.array(image), subtitle_text, (300, 600))  # Adjust position as needed
    #    ms.write_frame(image)
        renderer.render(image)
        frame += 1
        
    #ms.finalize()
    
    
    
#%%
        
