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
cam_man = lt.WebCam(cam_id=-1, shape_hw=shape_cam)
cam_man.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

use_maxperf = False
do_auto_acid = True

pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", local_files_only=True)
pipe.to("cuda")
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16, local_files_only=True)
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

prompt = 'dysfunctional insect family'
prompt = 'the garden of earthly delights by hieronymus bosch'
prompt = 'ocean waves'
prompt = 'powerful lightning discharges'
negative_prompt = "text, frame, photorealistic, photo"
prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)

list_prompts_all = []
with open("../coupling_diffusion/good_prompts.txt", "r") as file: 
    list_prompts_all = file.read().split('\n')

#%%  
n_steps = 30

# Image generation pipeline
sz = (512*2, 512*4)
renderer = lt.Renderer(width=sz[1], height=sz[0])
latents = torch.randn((1,4,64//1,64)).half().cuda()

cam_img = cam_man.get_img()
cam_img = np.flip(cam_img, axis=1)
last_image = np.uint8(cam_img)

akai_midimix = lt.MidiInput(device_name="akai_midimix")
memory_matrix = np.linspace(0.1,0.4,cam_img.shape[1])
memory_matrix = np.expand_dims(np.expand_dims(memory_matrix, 0), -1)
speech_detector = lt.Speech2Text()

noise_img2img_orig = torch.randn((1,4,75,100)).half().cuda()

# Iterate over blended prompts
while True:
    noise_img2img_fresh = torch.randn((1,4,75,100)).half().cuda()#randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    
    noise_mixing = akai_midimix.get("D0", val_min=0, val_max=1.0, val_default=0)
    noise_img2img = blender.interpolate_spherical(noise_img2img_orig, noise_img2img_fresh, noise_mixing)

    do_record_mic = akai_midimix.get("A3", button_mode="held_down")
    # do_record_mic = akai_lpd8.get('s', button_mode='pressed_once')
    
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
            
    get_motchie_prompt = akai_midimix.get('F4', button_mode='pressed_once')
    if get_motchie_prompt:
        prompt = np.random.choice(list_prompts_all)
        print(f"New prompt: {prompt}")
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)
        stop_recording = False        
    
    
    cam_img = cam_man.get_img()
    cam_img = np.flip(cam_img, axis=1)
    image = np.uint8(cam_img)
    acid_strength = akai_midimix.get("C0", val_min=0, val_max=1.0, val_default=0)
    strength = akai_midimix.get("C1", val_min=0.5, val_max=1.0, val_default=0.5)
    num_inference_steps = int(akai_midimix.get("C2", val_min=2, val_max=10, val_default=2))
    # mem_shift_speed = int(np.round(akai_midimix.get("D0", val_min=1, val_max=100, val_default=0)))
    # last_image = np.roll(last_image,mem_shift_speed,1)
    image = (1.-acid_strength)*image + acid_strength*last_image
    # image = np.uint8((1 - memory_matrix)*image + memory_matrix*last_image)
    # Generate the image using your pipeline
    image = pipe(image=Image.fromarray(image.astype(np.uint8)), 
                 latents=latents, num_inference_steps=num_inference_steps, strength=strength, 
                 guidance_scale=0.5, prompt_embeds=prompt_embeds, 
                 negative_prompt_embeds=negative_prompt_embeds, 
                 pooled_prompt_embeds=pooled_prompt_embeds, 
                 negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, noise_img2img=noise_img2img).images[0]
    last_image = np.array(image, dtype=float)
    
    do_antishift = akai_midimix.get("A4", button_mode="toggle")
    if do_antishift:
        last_image = np.roll(last_image,-2,axis=1)
    
    # last_image += np.random.randn(*last_image.shape) * noise_strength
    
    # Render the image
    renderer.render(image)
        
        
        
