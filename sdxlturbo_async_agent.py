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
import random as rn
import numpy as np
import asyncio

from prompt_blender import PromptBlender
from prompt_blender import PromptBlenderAsync

from gpt_agent import GPTAgent

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
pipe.vae = pipe.vae.cuda()
pipe.set_progress_bar_config(disable=True)


#%%
# load a premade assistant
assistant_id = 'asst_KrxchcAzg9rO4T0If7wwg2sx'
gpt = GPTAgent(assistant_id = assistant_id)

#%%

def prompt_processing(gpt, msg_prompt):
    message = gpt.send_message(msg_prompt)
    run = gpt.execute_run("")
    gpt.wait_for_run_completion()
    return gpt.get_all_replies()[0]
    

#%%

async def processed_prompt_input_loop(gpt, blender, update_event):
    while True:
        
        new_prompt = await asyncio.to_thread(input, "Enter new prompt: ")
        processed_prompt = prompt_processing(gpt, new_prompt)  # Process the new prompt
        blender.add_prompt(processed_prompt)  # Add the processed prompt
        update_event.set()  # Signal that a new prompt has been added

async def prompt_input_loop(blender, update_event):
    while True:
        new_prompt = await asyncio.to_thread(input, "Enter new prompt: ")
        blender.add_prompt(new_prompt)
        update_event.set()  # Signal that a new prompt has been added
        
        
async def render_loop(blender, pipe, update_event):
    sz = (512, 512)
    renderer = lt.Renderer(width=sz[1], height=sz[0])
    latents = torch.randn((1, 4, 64, 64)).half().cuda()

    current_index = 0  # Track the current position in the prompt sequence

    while True:processed_prompt_input_loop(blender, update_event))
    render_task = asyncio.create_task
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
            renderer.render(image.rotate(90))

            # Update the current index to move to the next prompt
            current_index += 1

        await asyncio.sleep(0.1)  # Short delay to yield control


async def main():
    initial_prompts = ["a pink and blue ocean seen from above", "a pink and blue ocean seen from above"]
    n_steps = 50
    blender = PromptBlenderAsync(pipe, initial_prompts, n_steps)
    update_event = asyncio.Event()
    

    # Start the asynchronous loops
    input_task = asyncio.create_task(processed_prompt_input_loop(gpt, blender, update_event))
    render_task = asyncio.create_task(render_loop(blender, pipe, update_event))

    # Run both tasks concurrently
    await asyncio.gather(input_task, render_task)

#%%

# Get the current event loop and run the main function
loop = asyncio.get_event_loop()
loop.run_until_complete(main())




