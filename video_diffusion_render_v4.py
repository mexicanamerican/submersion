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
from datasets import load_dataset
sys.path.append("../psychoactive_surface")
from prompt_blender import PromptBlender
from u_unet_modulated import forward_modulated
from tqdm import tqdm

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

shape_cam=(600,800) 
do_compile = False
use_community_prompts = True
use_modulated_unet = True
sz_renderwin = (512*2, 512*4)
resolution_factor = 5
base_w = 20
base_h = 15
# do_add_noise = True
negative_prompt = "blurry, bland, black and white, monochromatic"


#%% Aux Func and classes
class PromptManager: #change this to be more streamlined with the gpu device

    def __init__(self, use_community_prompts, prompts=None):
        self.use_community_prompts = use_community_prompts
        self.prompts = prompts
        self.current_prompt_index = 0  # To track the current index if prompts is a list
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
        if isinstance(self.prompts, str):  # Fixed prompt provided as a string
            return self.prompts
        elif isinstance(self.prompts, list):  # List of prompts provided
            prompt = self.prompts[self.current_prompt_index]
            self.current_prompt_index = (self.current_prompt_index + 1) % len(self.prompts)  # Loop through the list
            return prompt
        
        else:
            # Fallback to random prompt selection if no fixed or list of prompts provided
            if self.use_community_prompts:
                return random.choice(self.dataset['train'])['text']
            else:
                return random.choice(self.list_prompts_all)

    def save_harvested_prompt(self, prompt):
        with open(self.fp_save, "a", encoding="utf-8") as file:
            file.write(prompt + "\n")

            

#%% Inits



# Diffusion Pipe

gpu = "cuda:1"
device = torch.device("cuda:1")
pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_device=gpu, torch_dtype=torch.float16, variant="fp16", local_files_only=True)
pipe.to(gpu)
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device=gpu, torch_dtype=torch.float16, local_files_only=True)
pipe.vae = pipe.vae.cuda(gpu)
pipe.set_progress_bar_config(disable=True)


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

if use_modulated_unet:
    pipe.unet.forward = lambda *args, **kwargs: forward_modulated(pipe.unet, *args, **kwargs)
# Promptblender
blender = PromptBlender(pipe,1) # cuda device index 
# Renderer
renderer = lt.Renderer(width=sz_renderwin[1], height=sz_renderwin[0])

noise_resolution_w = base_w*resolution_factor
noise_resolution_h = base_h*resolution_factor

cam_resolution_w = 1920
cam_resolution_h = 1080
akai_midimix = lt.MidiInput(device_name="akai_midimix")
speech_detector = lt.Speech2Text()

# noise
latents = blender.get_latents()
noise_img2img_orig = torch.randn((1,4,noise_resolution_h,noise_resolution_w)).half().cuda(gpu)

image_displacement_accumulated = 0

#%% LOOP

modulations = {}
if use_modulated_unet:
    def noise_mod_func(sample):
        noise =  torch.randn(sample.shape, device=sample.device, generator=torch.Generator(device=sample.device).manual_seed(1))
        return noise    
    
    modulations['noise_mod_func'] = noise_mod_func
    
prompt_decoder = 'fire'
prompt_embeds_decoder, negative_prompt_embeds_decoder, pooled_prompt_embeds_decoder, negative_pooled_prompt_embeds_decoder = blender.get_prompt_embeds(prompt_decoder, negative_prompt)


#%%


video_list = ['EXTRA_SHOTS']

# Promptmanager
prompt_list = ['Astronauts, deep space, 4K, photorealistic', 'Cowboys, 4K, photorealistic', 'Miami 70s Fashion, Old TV show aesthetic', 'Funky Disco, analogue video aesthetic', 'Fitness Class, 4K, photorealistic', 'Indigenous people, black and white old video, documentary', 'Bull fighters, horse riding, 4K, photorealistic','People made of poop', 'Zombie horror scene, blood, 4K, bollywood']
promptmanager = PromptManager(False, prompt_list)


for video in video_list:
    
    print(f"Processing Video: {video}")
    mr = lt.MovieReader(video+'.mp4')
    ms = lt.MovieSaver(video+'PROCESSED2.mp4', fps=mr.fps_movie)
    #total_frames = mr.nmb_frames
    total_frames = mr.nmb_frames
    print(f"Number of frames: {total_frames}")
    prompt_change_interval = total_frames // len(promptmanager.prompts)  # Assuming promptmanager.prompts is a list of prompts
    
    nframe = np.asarray(Image.fromarray(mr.get_next_frame()).resize((cam_resolution_w,cam_resolution_h)))
    last_diffusion_image = np.uint8(nframe)
    last_cam_img_torch = None
    
    for frame_index in tqdm(range(total_frames)):
        
        do_fix_seed = not akai_midimix.get('F3', button_mode='toggle')
        if do_fix_seed:
            torch.manual_seed(0)
            
        noise_img2img_fresh = torch.randn((1,4,noise_resolution_h,noise_resolution_w)).half().cuda(gpu)#randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        
        noise_mixing = akai_midimix.get("D0", val_min=0, val_max=1.0, val_default=0)
        noise_img2img = blender.interpolate_spherical(noise_img2img_orig, noise_img2img_fresh, noise_mixing)
        do_add_noise = akai_midimix.get("G4", button_mode="toggle")
                
                
        if frame_index % prompt_change_interval == 0:
           # Change to the next prompt
           prompt = promptmanager.get_new_prompt()
           print(f"Current prompt: {prompt}")
           prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)
                
        cam_img = np.asarray(Image.fromarray(mr.get_next_frame()).resize((cam_resolution_w,cam_resolution_h)))
        cam_img = cv2.resize(cam_img.astype(np.uint8), (cam_resolution_w, cam_resolution_h))
    
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
            
        cam_img_torch = (1.-acid_strength)*cam_img_torch + acid_strength*torch_last_diffusion_image
        cam_img_torch = torch.clamp(cam_img_torch, 0, 255)
        cam_img = cam_img_torch.cpu().numpy()
            
        if use_modulated_unet:
            H2 = akai_midimix.get("H2", val_min=0, val_max=10, val_default=1)
            modulations['b0_samp'] = H2
            modulations['e2_samp'] = H2
            
            H1 = akai_midimix.get("H1", val_min=0, val_max=10, val_default=1)
            modulations['b0_emb'] = H1
            modulations['e2_emb'] = H1
            
            fract_mod = akai_midimix.get("G0", val_default=0, val_max=2, val_min=0)
            if fract_mod > 1:
                modulations['d*_extra_embeds'] = prompt_embeds_decoder    
            else:
                modulations['d*_extra_embeds'] = prompt_embeds
        else:
            modulations = None
            
        
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
                          modulations=modulations).images[0]
            
        last_diffusion_image = np.array(image, dtype=np.float32)
        
        do_antishift = akai_midimix.get("A4", button_mode="toggle")
        if do_antishift:
            last_diffusion_image = np.roll(last_diffusion_image,-4,axis=0)
        
        # Render the image
        renderer.render(image)
        ms.write_frame(image)
        
    ms.finalize()
        
        
        
