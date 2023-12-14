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

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

fract
#%%

@torch.no_grad()
def interpolate_spherical(p0, p1, fract_mixing: float):
    r"""
    Helper function to correctly mix two random variables using spherical interpolation.
    See https://en.wikipedia.org/wiki/Sllatents1erp
    The function will always cast up to float64 for sake of extra 4.
    Args:
        p0:
            First tensor for interpolation
        p1:
            Second tensor for interpolation
        fract_mixing: float
            Mixing coefficient of interval [0, 1].
            0 will return in p0
            1 will return in p1
            0.x will return a mix between both preserving angular velocity.
    """

    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'

    p0 = p0.double()
    p1 = p1.double()
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1 + epsilon, 1 - epsilon)

    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0 * s0 + p1 * s1

    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()

    return interp


#%%

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

#%% OLD
def blend_prompts(prompt_embeds1, negative_prompt_embeds1, pooled_prompt_embeds1, negative_pooled_prompt_embeds1, prompt_embeds2, negative_prompt_embeds2, pooled_prompt_embeds2, negative_pooled_prompt_embeds2, fract):

    prompt_embeds = interpolate_spherical(prompt_embeds1,prompt_embeds2,fract)
    negative_prompt_embeds = interpolate_spherical(negative_prompt_embeds1, negative_prompt_embeds2,fract)
    pooled_prompt_embeds = interpolate_spherical(pooled_prompt_embeds1, pooled_prompt_embeds2,fract)
    negative_pooled_prompt_embeds = interpolate_spherical(negative_pooled_prompt_embeds1, negative_pooled_prompt_embeds2,fract)
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

prompt_embeds1, negative_prompt_embeds1, pooled_prompt_embeds1, negative_pooled_prompt_embeds1 = get_prompt_embeds("photo of a house")
prompt_embeds2, negative_prompt_embeds2, pooled_prompt_embeds2, negative_pooled_prompt_embeds2 = get_prompt_embeds("photo of a submarine")

#%%
sz = (512, 512)
latents1 = torch.randn((1,4,64,64)).half().cuda()
latents2 = torch.randn((1,4,64,64)).half().cuda()

# prompt = "gore murder scene, photorealistic, 4K, cinema, visceral, disgusting, body parts"
# prompt = "Surface of the ocean on an alien planet with shades of light pink and blue"
renderer = lt.Renderer(width=sz[1], height=sz[0])
fract = 0
while True:
    fract += 0.009
    latents = latents1 #interpolate_spherical(latents1, latents2, fract)
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blend_prompts(prompt_embeds1, negative_prompt_embeds1, pooled_prompt_embeds1, negative_pooled_prompt_embeds1, prompt_embeds2, negative_prompt_embeds2, pooled_prompt_embeds2, negative_pooled_prompt_embeds2, fract)
    image = pipe(guidance_scale=0.0, num_inference_steps=1, latents=latents, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]#, output_type = 'np')[0][0,:,:,:]
    renderer.render(image.rotate(90))

#%%
def get_prompt_embeds(prompt):

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt="",
        negative_prompt_2="",
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        lora_scale=0,
        clip_skip=False,
    )
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

def blend_prompts(embeds1, embeds2, fract):
    """
    Blends two sets of prompt embeddings based on a specified fraction.
    """
    prompt_embeds1, negative_prompt_embeds1, pooled_prompt_embeds1, negative_pooled_prompt_embeds1 = embeds1
    prompt_embeds2, negative_prompt_embeds2, pooled_prompt_embeds2, negative_pooled_prompt_embeds2 = embeds2

    blended_prompt_embeds = interpolate_spherical(prompt_embeds1, prompt_embeds2, fract)
    blended_negative_prompt_embeds = interpolate_spherical(negative_prompt_embeds1, negative_prompt_embeds2, fract)
    blended_pooled_prompt_embeds = interpolate_spherical(pooled_prompt_embeds1, pooled_prompt_embeds2, fract)
    blended_negative_pooled_prompt_embeds = interpolate_spherical(negative_pooled_prompt_embeds1, negative_pooled_prompt_embeds2, fract)

    return (blended_prompt_embeds, blended_negative_prompt_embeds, blended_pooled_prompt_embeds, blended_negative_pooled_prompt_embeds)


def blend_sequence_prompts(prompts, n_steps):
    """
    Generates a sequence of blended prompt embeddings for a list of text prompts.
    """
    blended_prompts = []
    for i in range(len(prompts) - 1):
        prompt_embeds1 = get_prompt_embeds(prompts[i])
        prompt_embeds2 = get_prompt_embeds(prompts[i + 1])
        for step in range(n_steps):
            fract = step / float(n_steps - 1)
            blended = blend_prompts(prompt_embeds1, prompt_embeds2, fract)
            blended_prompts.append(blended)
    return blended_prompts

# Example usage
prompts = ["a man walking through the forest", "a man walking through the desert", "a man walking through the village", "a man walking through the war in the village","a man walking through the war in the village with explosions","a man walking through the destructed village, dead bodies, gore" , "a man walking through the desert", "a man walking through the forest"]
n_steps = 100
blended_prompts = blend_sequence_prompts(prompts, n_steps)



#%%

# Image generation pipeline
sz = (512, 512)
renderer = lt.Renderer(width=sz[1], height=sz[0])

# Iterate over blended prompts
for i in range(len(blended_prompts) - 1):
    fract = float(i) / (len(blended_prompts) - 1)
    blended = blend_prompts(blended_prompts[i], blended_prompts[i+1], fract)

    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blended

    # Calculate fract and blend latents
    latents = interpolate_spherical(latents1, latents2, fract)

    # Generate the image using your pipeline
    image = pipe(guidance_scale=0.0, num_inference_steps=1, latents=latents, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]

    # Render the image
    renderer.render(image.rotate(90))


