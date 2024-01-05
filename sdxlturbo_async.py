#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:13:24 2023

@author: lunar
"""

#%%
from diffusers import AutoPipelineForText2Image
import torch
import time

from diffusers import AutoencoderTiny
# this line breaks computers
# from sfast.compilers.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)
import xformers
import triton
import lunar_tools as lt
from PIL import Image
import numpy as np
from diffusers.utils.torch_utils import randn_tensor
from sfast.compilers.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)
import random as rn
import numpy as np
import asyncio

from prompt_blender import PromptBlender
from prompt_blender import PromptBlenderAsync

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

use_maxperf = True # 3 MIN COMPILE TIME  

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

    latents = torch.randn((1, 4, 96, 96)).half().cuda() # 64, 96
    current_index = 0
    initial_prompts = ["a pink and blue ocean seen from above", "a pink and blue ocean seen from above"]
    n_steps = 150
    fract = 0.5
    blender = PromptBlenderAsync(pipe, initial_prompts, n_steps)
    blended = blender.blend_prompts(blender.blended_prompts[current_index], blender.blended_prompts[current_index + 1], fract)
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blended
    image = pipe(guidance_scale=0.0, num_inference_steps=1, latents=latents, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]
    
    print('Compiled Successfully')

#%%

def prompt_processing(gpt4, msg_prompt):
    i_prompt = ["change the sentence 'an ocean seen from above' so as to portray the emotion of the prompt. for example, if you receive the prompt 'I am angry' you should output 'a red ocean seen from above'. only give the output - prompt: "]
    api_prompt = i_prompt[0] + msg_prompt[0]
    print(msg_prompt)
    processed_prompt = gpt4.generate(api_prompt)
    #print(processed_prompt)
    return processed_prompt

#%%

async def processed_prompt_input_loop(blender, update_event):
    while True:
        
        new_prompt = await asyncio.to_thread(input, "Enter new prompt: ")
        print(new_prompt)
        processed_prompt = prompt_processing(new_prompt)  # Process the new prompt
        print(processed_prompt)
        blender.add_prompt(processed_prompt)  # Add the processed prompt
        update_event.set()  # Signal that a new prompt has been added

async def prompt_input_loop(blender, update_event):
    while True:
        new_prompt = await asyncio.to_thread(input, "Enter new prompt: ")
        blender.add_prompt(new_prompt)
        update_event.set()  # Signal that a new prompt has been added
        
        
async def render_loop(blender, pipe, update_event):
    sz = (512*2, 512*2)
    renderer = lt.Renderer(width=sz[1], height=sz[0])
    latents = torch.randn((1, 4, 96, 96)).half().cuda() # 64, 96
    current_index = 0  # Track the current position in the prompt sequence

    while True:
        # Check if there's a new prompt to blend
        if update_event.is_set():
            update_event.clear()  # Reset the event
            # Ensure current index is within the new range of prompts
            current_index = min(current_index, len(blender.blended_prompts) - 2)

        # Blend from the current prompt to the next one
        if current_index < len(blender.blended_prompts) - 1:
            fract = 0.5  # Or any other logic to determine the blending fraction
            blended = blender.blend_prompts(blender.blended_prompts[current_index], blender.blended_prompts[current_index + 1], fract)

            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blended
            image = pipe(guidance_scale=0.0, num_inference_steps=1, latents=latents, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]
            renderer.render(image)

            # Update the current index to move to the next prompt
            current_index += 1

        await asyncio.sleep(0.1)  # Short delay to yield control


async def main():
    initial_prompts = ["a pink and blue ocean seen from above", "a pink and blue ocean seen from above"]
    n_steps = 100
    blender = PromptBlenderAsync(pipe, initial_prompts, n_steps)
    update_event = asyncio.Event()
    
    # Start the asynchronous loops
    input_task = asyncio.create_task(prompt_input_loop(blender, update_event))
    render_task = asyncio.create_task(render_loop(blender, pipe, update_event))

    # Run both tasks concurrently
    await asyncio.gather(input_task, render_task)


while True:
    
    initial_prompts = ["a pink and blue ocean seen from above", "a pink and blue ocean seen from above"]
    n_steps = 100
    blender = PromptBlenderAsync(pipe, initial_prompts, n_steps)
    
    # Start the asynchronous loops
    input_task = asyncio.create_task(prompt_input_loop(blender, update_event))
    render_task = asyncio.create_task(render_loop(blender, pipe, update_event))


# Get the current event loop and run the main function
#loop = asyncio.get_event_loop()
#loop.run_until_complete(main())




