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
cam_man = lt.WebCam(cam_id=0, shape_hw=shape_cam)
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
akai_lpd8 = lt.MidiInput("akai_lpd8")
blender = PromptBlender(pipe)

from nltk.corpus import wordnet as wn
nouns = list(wn.all_synsets('n'))
nouns = [nouns[i].lemma_names()[0] for i in range(len(nouns))]


prompt = 'photograph of elon musk, shot on analog film, kodak portra, high detail, iso 100, 16k, 4K Ultra HD, A lot of detail, best quality, masterpiece'
negative_prompt = 'blurry, cartoon, low resolution, wrong, bad quality, pixels'



# Image generation pipeline
sz = (512*2, 512*4)
renderer = lt.Renderer(width=sz[1], height=sz[0])
latents = torch.randn((1,4,64//1,64)).half().cuda()

prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)
# Iterate over blended prompts
while True:
    num_inference_steps = int(akai_lpd8.get("H1", val_min=2, val_max=5, val_default=2))
    strength = akai_lpd8.get("E0", val_min=0.7, val_max=1, val_default=0.5)
    guidance_scale = akai_lpd8.get("E1", val_min=0, val_max=1, val_default=0.5)
    cam_img = cam_man.get_img()
    cam_img = np.flip(cam_img, axis=1)
    image = Image.fromarray(np.uint8(cam_img))
    
    # Generate the image using your pipeline
    image = pipe(image=image, latents=latents, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=0.5, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]

    # Render the image
    renderer.render(image)
    
        
        
