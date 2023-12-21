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
import numpy as np
from PIL import Image

from prompt_blender import PromptBlender

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

image = load_image("https://w0.peakpx.com/wallpaper/123/793/HD-wallpaper-two-worlds-art-fantasy-sky-woman-landscape.jpg").resize((512, 512))
    
prompts = []
prompts.append('Esmeralda and John have met in castle made of dust. They are having a conversation.')
prompts.append('Esmeralda and John have met in castle made of dust. They are having a conversation. They are thinking of getting married')
prompts.append('The couple is visiting. Old house in the village. Year 1950')
prompts.append('Esmeralda and John have met in castle made of dust. They are having a conversation. They discuss their wedding attires')
prompts.append('Esmeralda and John have met in castle made of dust. They are having a conversation. They are now in the desert')
    
n_steps = 50
blended_prompts = blender.blend_sequence_prompts(prompts, n_steps)

# Image generation pipeline
sz = (512*1, 512*2)
renderer = lt.Renderer(width=sz[1], height=sz[0])
latents = torch.randn((1,4,64//1,64)).half().cuda()

noise_level = 5

# Iterate over blended prompts
while True:
    print('restarting...')
    
    for i in range(len(blended_prompts) - 1):
        fract = float(i) / (len(blended_prompts) - 1)
        blended = blender.blend_prompts(blended_prompts[i], blended_prompts[i+1], fract)
    
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blended
        
        if use_image_mode:
            # add noise and shift
            img_array = np.array(image)
            shifted_array = np.roll(img_array, -3, axis=1)
            shifted_array[:, -1] = 0
            noise = np.random.uniform(-noise_level, noise_level, shifted_array.shape)
            shifted_array = shifted_array + noise
            shifted_array = np.clip(shifted_array, 0, 255)        
            image = Image.fromarray(shifted_array.astype('uint8'))
        
        if use_image_mode:
            image = pipe(image=image, latents=latents, num_inference_steps=2, strength=0.5, guidance_scale=0.5, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]
        else:
            image = pipe(guidance_scale=0.0, num_inference_steps=1, latents=latents, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]
        
    
        # Render the image
        renderer.render(image)
        
        
        
