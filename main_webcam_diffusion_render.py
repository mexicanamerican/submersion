#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
possibly short term
- smooth prompt blending A -> B
- automatic prompt injection
- investigate better noise
- understand mem acid better
- smooth continuation mode
- objects floating around or being interactive

nice for cosyne
- physical objects

long term
- parallelization and stitching
"""



#%%`
import sys
sys.path.append('../')


from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
import torch
import time

from diffusers import AutoencoderTiny

from sfast.compilers.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)
from diffusers.utils import load_image
import random
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
import sys
# from datasets import load_dataset
sys.path.append("../psychoactive_surface")
from prompt_blender import PromptBlender
from u_unet_modulated import forward_modulated
import os
from dotenv import load_dotenv #pip install python-dotenv
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

#%% VARS
shape_cam=(600,800) 
do_compile = False
use_community_prompts = False
use_modulated_unet = True
sz_renderwin = (512*2, 512*4)
resolution_factor = 5
base_w = 20
base_h = 15
do_add_noise = True
negative_prompt = "blurry, bland, black and white, monochromatic"

# load models
# To use the models in offline mode, ensure you have a .env file in your project directory.
# This file should contain the paths to your local models using the keys MODEL_TURBO and MODEL_VAE.
# Example of .env content:
# MODEL_TURBO=path/to/your/local/model_turbo
# MODEL_VAE=path/to/your/local/model_vae
load_dotenv()
env_model_turbo = os.getenv("MODEL_TURBO")
if env_model_turbo:
    model_turbo = env_model_turbo
    print(f"Using local model for model_turbo: {model_turbo}")
else:
    model_turbo = "stabilityai/sdxl-turbo"

env_model_vae = os.getenv("MODEL_VAE")
if env_model_vae:
    model_vae = env_model_vae
    print(f"Using local model for VAE: {model_vae}")
else:
    model_vae = "madebyollin/taesdxl"
#%% Aux Func and classes
class PromptManager:
    def __init__(self, use_community_prompts):
        self.use_community_prompts = use_community_prompts
        self.hf_dataset = "FredZhang7/stable-diffusion-prompts-2.47M"
        self.local_prompts_path = "../psychoactive_surface/good_prompts.txt"
        self.fp_save = "good_prompts_harvested.txt"
        if self.use_community_prompts:
            self.dataset = load_dataset(self.hf_dataset)
        else:
            self.list_prompts_all = self.load_local_prompts()

    def load_local_prompts(self):
        with open(self.local_prompts_path, "r", encoding="utf-8") as file:
            return file.read().split('\n')

    def get_new_prompt(self):
        if self.use_community_prompts:
            return random.choice(self.dataset['train'])['text']
        else:
            return random.choice(self.list_prompts_all)

    def save_harvested_prompt(self, prompt):
        with open(self.fp_save, "a", encoding="utf-8") as file:
            file.write(prompt + "\n")
            

#%% Inits
cam = lt.WebCam(cam_id=-1, shape_hw=shape_cam)
cam.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)

# Diffusion Pipe
pipe = AutoPipelineForImage2Image.from_pretrained(model_turbo, torch_dtype=torch.float16, variant="fp16", local_files_only=True)
pipe.to("cuda")
pipe.vae = AutoencoderTiny.from_pretrained(model_vae, torch_device='cuda', torch_dtype=torch.float16, local_files_only=True)
pipe.vae = pipe.vae.cuda()
pipe.set_progress_bar_config(disable=True)

if use_modulated_unet:
    #pipe.unet.forward = forward_modulated
    pipe.unet.forward = forward_modulated.__get__(pipe.unet, UNet2DConditionModel)
    # pipe.unet.forward = lambda *args, **kwargs: forward_modulated(pipe.unet, *args, **kwargs)
    
if do_compile:
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

# Promptblender
blender = PromptBlender(pipe)

# Promptmanager
promptmanager = PromptManager(use_community_prompts)
prompt = promptmanager.get_new_prompt()

prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)

# Renderer
renderer = lt.Renderer(width=sz_renderwin[1], height=sz_renderwin[0])

cam_img = cam.get_img()
cam_img = np.flip(cam_img, axis=1)


noise_resolution_w = base_w*resolution_factor
noise_resolution_h = base_h*resolution_factor

cam_resolution_w = base_w*8*resolution_factor
cam_resolution_h = base_h*8*resolution_factor

# test resolution
cam_img = cv2.resize(cam_img.astype(np.uint8), (cam_resolution_w, cam_resolution_h))

# fp_aug = 'augs/baloon.png'
# aug_overlay = cv2.imread(fp_aug)[:,:,::-1].copy()
# aug_overlay = cv2.resize(aug_overlay.astype(np.uint8), (cam_resolution_w, cam_resolution_h))

last_diffusion_image = np.uint8(cam_img)
last_cam_img_torch = None

akai_midimix = lt.MidiInput(device_name="akai_midimix")
memory_matrix = np.linspace(0.1,0.4,cam_img.shape[1])
memory_matrix = np.expand_dims(np.expand_dims(memory_matrix, 0), -1)
speech_detector = lt.Speech2Text()

# noise
latents = blender.get_latents()
noise_img2img_orig = torch.randn((1,4,noise_resolution_h,noise_resolution_w)).half().cuda()

image_displacement_accumulated = 0

def get_sample_shape_unet(coord):
    if coord[0] == 'e':
        coef = float(2**int(coord[1]))
        shape = [int(np.ceil(noise_resolution_h/coef)), int(np.ceil(noise_resolution_w/coef))]
    elif coord[0] == 'b':
        shape = [int(np.ceil(noise_resolution_h/4)), int(np.ceil(noise_resolution_w/4))]
    else:
        coef = float(2**(2-int(coord[1])))
        shape = [int(np.ceil(noise_resolution_h/coef)), int(np.ceil(noise_resolution_w/coef))]
        
    return shape

#%% LOOP

def get_noise_for_modulations(shape):
    return torch.randn(shape, device=pipe.device, generator=torch.Generator(device=pipe.device).manual_seed(1)).half()

modulations = {}
if use_modulated_unet:
    modulations_noise = {}
    for i in range(3):
        modulations_noise[f'e{i}'] = get_noise_for_modulations(get_sample_shape_unet(f'e{i}'))
        modulations_noise[f'd{i}'] = get_noise_for_modulations(get_sample_shape_unet(f'd{i}'))
        
    modulations_noise['b0'] = get_noise_for_modulations(get_sample_shape_unet('b0'))
    
prompt_decoder = 'fire'
prompt_embeds_decoder, negative_prompt_embeds_decoder, pooled_prompt_embeds_decoder, negative_pooled_prompt_embeds_decoder = blender.get_prompt_embeds(prompt_decoder, negative_prompt)

last_render_timestamp = time.time()

while True:
    
    do_fix_seed = not akai_midimix.get('F3', button_mode='toggle')
    if do_fix_seed:
        torch.manual_seed(0)
        
    noise_img2img_fresh = torch.randn((1,4,noise_resolution_h,noise_resolution_w)).half().cuda()#randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    
    noise_mixing = akai_midimix.get("D0", val_min=0, val_max=1.0, val_default=0)
    noise_img2img = blender.interpolate_spherical(noise_img2img_orig, noise_img2img_fresh, noise_mixing)
    # do_add_noise = akai_midimix.get("G4", button_mode="toggle")
    do_record_mic = akai_midimix.get("A3", button_mode="held_down")
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
            
    get_new_prompt = akai_midimix.get('B3', button_mode='pressed_once')
    if get_new_prompt:
        try:
            prompt = promptmanager.get_new_prompt()
            print(f"New prompt: {prompt}")
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)
            stop_recording = False
        except Exception as e:
            print(f"fail! {e}")

    save_prompt = akai_midimix.get('B4', button_mode='pressed_once')
    if save_prompt:
        promptmanager.save_harvested_prompt(prompt)
            
    cam_img = cam.get_img()
    cam_img = np.flip(cam_img, axis=1)
    
    # test resolution
    cam_img = cv2.resize(cam_img.astype(np.uint8), (cam_resolution_w, cam_resolution_h))
    
    # do_aug_overlay = akai_midimix.get('C3', button_mode='toggle')
    # if do_aug_overlay:
    #     aug_overlay = np.roll(aug_overlay,-10, axis=0)
    #     mask_aug = aug_overlay[:,:,0] != 0
    #     cam_img[mask_aug] = aug_overlay[mask_aug]
    
    strength = akai_midimix.get("C1", val_min=0.5, val_max=1.0, val_default=0.5)
    num_inference_steps = int(akai_midimix.get("C2", val_min=2, val_max=10, val_default=2))
    
    cam_img_torch = torch.from_numpy(cam_img.copy()).to(latents.device).float()
    apply_quant = akai_midimix.get("G3", button_mode="toggle")
    torch_last_diffusion_image = torch.from_numpy(last_diffusion_image).to(cam_img_torch)
    if do_add_noise:
        # coef noise
        coef_noise = akai_midimix.get("E0", val_min=0, val_max=0.3, val_default=0.05)
        t_rand = (torch.rand(cam_img_torch.shape, device=cam_img_torch.device)[:,:,0].unsqueeze(2) - 0.5) * coef_noise * 255 * 5
        cam_img_torch += t_rand
        torch_last_diffusion_image += t_rand
        # cam_img_torch += (torch.rand(cam_img_torch.shape, device=cam_img_torch.device)[:,:,0].unsqueeze(2) - 0.5) * coef_noise * 255 * 5
    
    if apply_quant:
        ## quantization
        quant_strength = int(akai_midimix.get("G2", val_min=1, val_max=100))
        cam_img_torch = (cam_img_torch/quant_strength).floor() * quant_strength

    do_accumulate_acid = akai_midimix.get("C4", button_mode="toggle")

    if do_accumulate_acid:
        ## displacement controlled acid
        if last_cam_img_torch is None:
            last_cam_img_torch = cam_img_torch
            
        image_displacement_array = ((cam_img_torch - last_cam_img_torch)/255)**2

        image_displacement = image_displacement_array.mean()
        acid_gain = akai_midimix.get("C0", val_min=0, val_max=1.0, val_default=0.05)
        image_displacement = (1-image_displacement*100)
        if image_displacement < 0:
            image_displacement = 0
            
        if image_displacement >= 0.5:
            image_displacement_accumulated += 2e-2
        else:
            image_displacement_accumulated -= 2e-1
            
        if image_displacement_accumulated < 0:
            image_displacement_accumulated = 0
            
        acid_strength = 0.1 + image_displacement_accumulated * 1
        acid_strength *= acid_gain
        last_cam_img_torch = cam_img_torch.clone()
    else:
        acid_strength = akai_midimix.get("C0", val_min=0, val_max=1.0, val_default=0.05)
        
    # just a test
    # cam_img_torch = (1.-acid_strength)*cam_img_torch + acid_strength*torch.from_numpy(last_diffusion_image).to(cam_img_torch)
    cam_img_torch = (1.-acid_strength)*cam_img_torch + acid_strength*torch_last_diffusion_image
    # if akai_midimix.get('E4', button_mode='pressed_once'):
    #     xxx
    cam_img_torch = torch.clamp(cam_img_torch, 0, 255)
    cam_img = cam_img_torch.cpu().numpy()
        
    if use_modulated_unet:
        H2 = akai_midimix.get("H2", val_min=0, val_max=10, val_default=1)
        modulations['b0_samp'] = torch.tensor(H2, device=latents.device)
        modulations['e2_samp'] = torch.tensor(H2, device=latents.device)
        
        H1 = akai_midimix.get("H1", val_min=0, val_max=10, val_default=1)
        modulations['b0_emb'] = torch.tensor(H1, device=latents.device)
        modulations['e2_emb'] = torch.tensor(H1, device=latents.device)
        
        fract_mod = akai_midimix.get("G0", val_default=0, val_max=2, val_min=0)
        if fract_mod > 1:
            modulations['d*_extra_embeds'] = prompt_embeds_decoder    
        else:
            modulations['d*_extra_embeds'] = prompt_embeds
            
        modulations['modulations_noise'] = modulations_noise
    else:
        modulations = None
        
    if use_modulated_unet:
        cross_attention_kwargs ={}
        cross_attention_kwargs['modulations'] = modulations
    else:
        cross_attention_kwargs = None
    
    use_debug_overlay = akai_midimix.get("H3", button_mode="toggle")
    if use_debug_overlay:
        image = cam_img.astype(np.uint8)
    else:
        image = pipe(image=Image.fromarray(cam_img.astype(np.uint8)), 
                      latents=latents, num_inference_steps=num_inference_steps, strength=strength, 
                      guidance_scale=0.5, prompt_embeds=prompt_embeds, 
                      negative_prompt_embeds=negative_prompt_embeds, 
                      pooled_prompt_embeds=pooled_prompt_embeds, 
                      negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, noise_img2img=noise_img2img, 
                      modulations=modulations,cross_attention_kwargs=cross_attention_kwargs).images[0]
        
    time_difference = time.time() - last_render_timestamp
    last_render_timestamp = time.time()
    
    print(f'fps: {np.round(1/time_difference)}')
    
    last_diffusion_image = np.array(image, dtype=np.float32)
    
    do_antishift = akai_midimix.get("A4", button_mode="toggle")
    if do_antishift:
        last_diffusion_image = np.roll(last_diffusion_image,-4,axis=0)
    
    # Render the image
    renderer.render(image)
        
        
        
