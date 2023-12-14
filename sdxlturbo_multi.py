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
#from sfast.compilers.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)
import xformers
import triton
import lunar_tools as lt
from PIL import Image
import numpy as np
from diffusers.utils.torch_utils import randn_tensor
import random as rn
import numpy as np

from prompt_blender import PromptBlender

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


#%%
# config = CompilationConfig.Default()

# config.enable_xformers = True
# config.enable_triton = True

# config.enable_cuda_graph = True

# config.enable_jit = True
# config.enable_jit_freeze = True
# config.trace_scheduler = True
# config.enable_cnn_optimization = True
# config.preserve_parameters = False
# config.prefer_lowp_gemm = True

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")

pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
pipe.vae = pipe.vae.cuda()

pipe.set_progress_bar_config(disable=True)

# pipe = compile(pipe, config)

#%%

# Example usage
blender = PromptBlender(pipe)
prompts = ["a man walking through the forest", "a man walking through the desert", "a man walking through the village", "a man walking through the war in the village","a man walking through the war in the village with explosions","a man walking through the destructed village, dead bodies, gore" , "a man walking through the desert", "a man walking through the forest"]
prompts = ["a cat", "a dog", "a bird", "a fish","a whale"]
prompts = ["photorealistic gore scene nsfw, blood", 'photorealistic gore scene nsfw, blood, multiple people, horror']


a = 1
b = 101
prompts = [
    f'scene: a green field, detail: clouds {nmb}'
    for nmb in [rn.randint(a, b) for _ in range(100)]
] + [
    f'scene: a green field, detail: flowers {nmb}'
    for nmb in [rn.randint(a, b) for _ in range(100)]
]


#prompts = [f'{nmb} arm' for nmb in range(100)]
# prompts = ["a cat eating", "a cat running", "a cat sleeping", "a cat snoring"]
n_steps = 10
blended_prompts = blender.blend_sequence_prompts(prompts, n_steps)


# Image generation pipeline
sz = (512, 512)
renderer = lt.Renderer(width=sz[1], height=sz[0])
latents = torch.randn((1,4,64,64)).half().cuda()

# Iterate over blended prompts
for i in range(len(blended_prompts) - 1):
    fract = float(i) / (len(blended_prompts) - 1)
    blended = blender.blend_prompts(blended_prompts[i], blended_prompts[i+1], fract)

    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blended

    # Generate the image using your pipeline
    image = pipe(guidance_scale=0.0, num_inference_steps=1, latents=latents, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]

    # Render the image
    renderer.render(image.rotate(90))


