#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
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

#%%

shape_cam=(600,800) 
do_compile = True
use_community_prompts = True
width_latents = 96
height_latents = 64


#%% Inits


cam = lt.WebCam(cam_id=-1, shape_hw=shape_cam)
cam.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", local_files_only=True)
pipe.to("cuda")
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16, local_files_only=True)
pipe.vae = pipe.vae.cuda()
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

#%%
blender = PromptBlender(pipe)
blender.w = width_latents
blender.h = height_latents

#%%
class Deflicker:
    def __init__(self):
        self.init_kernel()
        self.threshold = 5
        
    def init_kernel(self, ksize=3):
        x, y = np.meshgrid(np.linspace(-1,1,ksize), np.linspace(-1,1,ksize))
        d = np.sqrt(x*x+y*y)
        sigma, mu = 1.0, 0.0
        g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        g = g/np.sum(g)
        kernel_weights = g
        padding = int((kernel_weights.shape[0]-1) / 2)
        m = torch.nn.Conv2d(1, 1, kernel_weights.shape[0], padding=padding, stride=1)
        m.bias[0] = 0
        m.weight[0,0,:,:] = torch.from_numpy(kernel_weights.astype(np.float32))
        m = m.cuda()
        self.kernel = m

    def apply_kernel(self, img):
        if len(img.shape)==2:
            img = img.expand(1,1,img.shape[0],img.shape[1])
        else:
            print("WARNING! 3D NEEDS 3D KERNEL!")
            img = img.permute([2,0,1])
            img = img.expand(1,img.shape[0],img.shape[1],img.shape[2])
            
        img = self.kernel(img)
            
        return img.squeeze()
        
    def apply(self, last_frame, current_frame, return_numpy=False):
        current_frame_avg = current_frame.mean(axis=-1)
        last_frame_avg = last_frame.mean(axis=-1)
        
        # Apply convolution to both frames
        filtered_last_frame = self.apply_kernel(current_frame_avg)
        filtered_current_frame = self.apply_kernel(last_frame_avg)
        
        # Calculate absolute change and create a mask for changes above the threshold
        abs_change = torch.abs(filtered_current_frame - filtered_last_frame)
        change_mask = abs_change > self.threshold
        change_mask = change_mask.unsqueeze(-1).repeat(1, 1, 3)
        
        # Apply the change mask
        new_current_frame = torch.where(change_mask, current_frame, last_frame)

        # apply_pullback = akai_midimix.get("F4:", button_mode="toggle")
        # pull_strength = akai_midimix.get("F1", val_min=0, val_max=0.1, val_default=0.01)
        # if apply_pullback:
        #     direction = (current_frame - last_frame) * pull_strength
        #     new_current_frame += direction

        
        # Optionally return as a numpy array
        if return_numpy:
            new_current_frame = new_current_frame.cpu().numpy()
        
        
        return new_current_frame

deflicker = Deflicker()

#%% Prompts
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



promptmanager = PromptManager(use_community_prompts)
prompt = promptmanager.get_new_prompt()
negative_prompt = "blurry, bland"
prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)
#%%  
n_steps = 30

# Image generation pipeline
sz = (512*2, 512*4)
renderer = lt.Renderer(width=sz[1], height=sz[0])
latents = blender.get_latents()
# latents = torch.randn((1,4,64//1,64)).half().cuda()

cam_img = cam.get_img()
cam_img = np.flip(cam_img, axis=1)

resolution_factor = 8

base_w = 20
base_h = 15

noise_resolution_w = base_w*resolution_factor
noise_resolution_h = base_h*resolution_factor

cam_resolution_w = base_w*8*resolution_factor
cam_resolution_h = base_h*8*resolution_factor

# test resolution
cam_img = cv2.resize(cam_img.astype(np.uint8), (cam_resolution_w, cam_resolution_h))

last_diffusion_image = np.uint8(cam_img)
last_cam_img_torch = None

akai_midimix = lt.MidiInput(device_name="akai_midimix")
memory_matrix = np.linspace(0.1,0.4,cam_img.shape[1])
memory_matrix = np.expand_dims(np.expand_dims(memory_matrix, 0), -1)
speech_detector = lt.Speech2Text()

noise_img2img_orig = torch.randn((1,4,noise_resolution_h,noise_resolution_w)).half().cuda()

image_displacement_accumulated = 0





# Iterate over blended prompts
while True:
    
    torch.manual_seed(0)
    
    noise_img2img_fresh = torch.randn((1,4,noise_resolution_h,noise_resolution_w)).half().cuda()#randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    
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
            
    get_new_prompt = akai_midimix.get('B3', button_mode='pressed_once')
    if get_new_prompt:
        prompt = promptmanager.get_new_prompt()
        print(f"New prompt: {prompt}")
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)
        stop_recording = False

    save_prompt = akai_midimix.get('B4', button_mode='pressed_once')
    if save_prompt:
        promptmanager.save_harvested_prompt(prompt)
            
    cam_img = cam.get_img()
    cam_img = np.flip(cam_img, axis=1)
    
    # test resolution
    cam_img = cv2.resize(cam_img.astype(np.uint8), (cam_resolution_w, cam_resolution_h))
    
    strength = akai_midimix.get("C1", val_min=0.5, val_max=1.0, val_default=0.5)
    num_inference_steps = int(akai_midimix.get("C2", val_min=2, val_max=10, val_default=2))
    
    do_auto_acid = akai_midimix.get("E3", button_mode="toggle")

    if do_auto_acid:
        cam_img_torch = torch.from_numpy(cam_img.copy()).to(latents.device).float()

        apply_quant = akai_midimix.get("G3", button_mode="toggle")
        quant_strength = int(akai_midimix.get("G2", val_min=1, val_max=100))

        if apply_quant:
            cam_img_torch = (cam_img_torch/quant_strength).floor() * quant_strength


        if last_cam_img_torch is None:
            last_cam_img_torch = cam_img_torch
            
            
        # Deflicker
        apply_deflicker = akai_midimix.get("F3", button_mode="toggle")
        if apply_deflicker:
            deflicker_strength = akai_midimix.get("F0", val_min=0, val_max=200)
            deflicker.threshold = deflicker_strength
            cam_img_torch = deflicker.apply(last_cam_img_torch, cam_img_torch)
            




        # cam_img_torch_blur = cam_img_torch.unsqueeze(0).permute([0,3,1,2])
        # cam_img_torch_blur = torch.nn.functional.interpolate(cam_img_torch_blur, (150,200))
        # cam_img_torch_blur = torch.nn.functional.interpolate(cam_img_torch_blur, (600,800))
        # cam_img_torch = cam_img_torch_blur[0].permute([1,2,0])            
        
        image_displacement_array = ((cam_img_torch - last_cam_img_torch)/255)**2


        image_displacement = image_displacement_array.mean()
        acid_gain = akai_midimix.get("C0", val_min=0, val_max=1.0, val_default=0)
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
        # if acid_strength > 0.4:
        #     acid_strength =  0.4
        # schleifing
        coef_schleifing = akai_midimix.get("E0", val_min=0, val_max=0.2, val_default=0)
        
        # image_displacement_array_blur = image_displacement_array.unsqueeze(0).permute([0,3,1,2])
        # image_displacement_array_blur = torch.nn.functional.interpolate(image_displacement_array_blur, (60,80))
        # image_displacement_array_blur = torch.nn.functional.interpolate(image_displacement_array_blur, (600,800))
        # image_displacement_array = image_displacement_array_blur[0].permute([1,2,0])
        
        mask_schleifing = (image_displacement_array.max(dim=2)[0] < coef_schleifing)
        
        # do_decay_schleifing = akai_midimix.get("B4", button_mode="toggle")
        # if do_decay_schleifing:
        #     cam_img_torch[mask_schleifing] = last_cam_img_torch[mask_schleifing]*0.8 + cam_img_torch[mask_schleifing]*0.2
        # else:
        #     cam_img_torch[mask_schleifing] = last_cam_img_torch[mask_schleifing]
        #     cam_img_torch[mask_schleifing] = last_cam_img_torch[mask_schleifing]
        
        center_noise = akai_midimix.get("E4", button_mode="toggle")
        if center_noise:
            cam_img_torch += (torch.rand(cam_img_torch.shape, device=cam_img_torch.device)[:,:,0].unsqueeze(2) - 0.5) * coef_schleifing * 255 * 5
        else:
            cam_img_torch += torch.rand(cam_img_torch.shape, device=cam_img_torch.device)[:,:,0].unsqueeze(2) * coef_schleifing * 255 * 5
    
        # cam_img_torch += torch.rand(cam_img_torch.shape, device=cam_img_torch.device)[:,:,0].unsqueeze(2) * coef_schleifing * 255 * 5
            
        #cam_img_torch = mask_schleifing.unsqueeze(2).repeat([1,1,3]).float() * 255
        
        acid_strength *= acid_gain
        
        # print(f'acid_strength {acid_strength} image_displacement_accumulated {image_displacement_accumulated}')
        
        cam_img_torch = torch.clamp(cam_img_torch, 0, 255)
        cam_img = cam_img_torch.cpu().numpy()
        
        #print(f'max cam_img {cam_img.max()}')
        last_cam_img_torch = cam_img_torch.clone()
    else:
        acid_strength = akai_midimix.get("C0", val_min=0, val_max=1.0, val_default=0)
        
    # mem_shift_speed = int(np.round(akai_midimix.get("D0", val_min=1, val_max=100, val_default=0)))
    # last_image = np.roll(last_image,mem_shift_speed,1)
    
    cam_img = (1.-acid_strength)*cam_img.astype(np.float32) + acid_strength*last_diffusion_image
    
    # image = np.uint8((1 - memory_matrix)*image + memory_matrix*last_image)
    # Generate the image using your pipeline
    
    use_debug_overlay = akai_midimix.get("H3", button_mode="toggle")
    
    if use_debug_overlay:
        image = cam_img.astype(np.uint8)
    else:
        image = pipe(image=Image.fromarray(cam_img.astype(np.uint8)), 
                      latents=latents, num_inference_steps=num_inference_steps, strength=strength, 
                      guidance_scale=0.5, prompt_embeds=prompt_embeds, 
                      negative_prompt_embeds=negative_prompt_embeds, 
                      pooled_prompt_embeds=pooled_prompt_embeds, 
                      negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, noise_img2img=noise_img2img).images[0]
        
    last_diffusion_image = np.array(image, dtype=np.float32)
    
    do_antishift = akai_midimix.get("A4", button_mode="toggle")
    if do_antishift:
        last_diffusion_image = np.roll(last_diffusion_image,-2,axis=1)
    
    # last_image += np.random.randn(*last_image.shape) * noise_strength
    
    # Render the image
    renderer.render(image)
        
        
        
