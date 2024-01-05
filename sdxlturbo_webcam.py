#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:13:24 2023

@author: lunar
"""

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

from prompt_blender import PromptBlender

shape_cam=(600,800) 
cam_man = lt.WebCam(cam_id=7, shape_hw=shape_cam)
cam_man.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

use_maxperf = False

pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
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
# Example usage
blender = PromptBlender(pipe)

from nltk.corpus import wordnet as wn
nouns = list(wn.all_synsets('n'))
nouns = [nouns[i].lemma_names()[0] for i in range(len(nouns))]

# base = 'skeleton person head skull terrifying'
base = 'very bizarre and grotesque zombie monster'

tp = 150
prompts = []
for i in range(tp):
    prompts.append(f'A painting of {base} who looks like {nouns[np.random.randint(len(nouns))]}. 4K Ultra HD. A lot of detail')

n_steps = 30
blended_prompts = blender.blend_sequence_prompts(prompts, n_steps)

# Image generation pipeline
sz = (512*1, 512*2)
renderer = lt.Renderer(width=sz[1], height=sz[0])
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
        image = Image.fromarray(np.uint8(cam_img))
        
        # Generate the image using your pipeline
        image = pipe(image=image, latents=latents, num_inference_steps=2, strength=0.999, guidance_scale=0.5, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]
    
        # Render the image
        renderer.render(image)
        
        
        
