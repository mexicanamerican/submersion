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
# import random as rn
import numpy as np
import xformers
import triton
import cv2
import sys
from datasets import load_dataset
sys.path.append("../psychoactive_surface")
from prompt_blender import PromptBlender
from u_unet_modulated import forward_modulated
import os
from dotenv import load_dotenv #pip install python-dotenv
from kornia.filters.kernels import get_binary_kernel2d
from typing import List, Tuple
from human_seg import HumanSeg

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False 

# key keys: H3 ->  B3/A3 -> E0 (noise) -> C0 (acid) -> A4 (B0+B1)
#%% VARS
# shape_cam=(600,800) 
shape_cam=(300,400) 
do_compile = True
use_community_prompts = False

sz_renderwin = (512*2, int(512*2*16/9))
resolution_factor = 8
base_w = 20
base_h = 15


resolution_factor = 7 # native is 4
base_w = 16
base_h = 16

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
        self.hf_dataset = "Gustavosta/Stable-Diffusion-Prompts"
        # self.hf_dataset = "FredZhang7/stable-diffusion-prompts-2.47M"
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
            try:
                return random.choice(self.dataset['train'])['text']
            except:
                return random.choice(self.dataset['train'])['Prompt']
        else:
            return random.choice(self.list_prompts_all)

    def save_harvested_prompt(self, prompt):
        with open(self.fp_save, "a", encoding="utf-8") as file:
            file.write(prompt + "\n")
            
import torch.nn.functional as F

def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    computed: List[int] = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]


def median_blur(input: torch.Tensor,
                kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Blurs an image using the median filter.

    Args:
        input (torch.Tensor): the input image with shape :math:`(B,C,H,W)`.
        kernel_size (Tuple[int, int]): the blurring kernel size.

    Returns:
        torch.Tensor: the blurred input tensor with shape :math:`(B,C,H,W)`.

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> output = median_blur(input, (3, 3))
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    padding: Tuple[int, int] = _compute_zero_padding(kernel_size)

    # prepare kernel
    kernel: torch.Tensor = get_binary_kernel2d(kernel_size).to(input)
    b, c, h, w = input.shape

    # map the local window to single vector
    features: torch.Tensor = F.conv2d(
        input.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1)
    features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

    # compute the median along the feature axis
    median: torch.Tensor = torch.median(features, dim=2)[0]

    return median

class MedianBlur(nn.Module):
    r"""Blurs an image using the median filter.

    Args:
        kernel_size (Tuple[int, int]): the blurring kernel size.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = MedianBlur((3, 3))
        >>> output = blur(input)
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """

    def __init__(self, kernel_size: Tuple[int, int]) -> None:
        super(MedianBlur, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return median_blur(input, self.kernel_size)

def zoom_image_torch(input_tensor, zoom_factor):
    # Ensure the input is a 4D tensor [batch_size, channels, height, width]
    input_tensor = input_tensor.permute(2,0,1)
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
    
    # Original size
    original_height, original_width = input_tensor.shape[2], input_tensor.shape[3]
    
    # Calculate new size
    new_height = int(original_height * zoom_factor)
    new_width = int(original_width * zoom_factor)
    
    # Interpolate
    zoomed_tensor = F.interpolate(input_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
    # zoomed_tensor = F.interpolate(input_tensor, size=(new_width, new_height), mode='bilinear', align_corners=False).permute(1,0,2)
    
    # Calculate padding to match original size
    pad_height = (original_height - new_height) // 2
    pad_width = (original_width - new_width) // 2
    
    # Adjust for even dimensions to avoid negative padding
    pad_height_extra = original_height - new_height - 2*pad_height
    pad_width_extra = original_width - new_width - 2*pad_width
    
    # Pad to original size
    if zoom_factor < 1:
        zoomed_tensor = F.pad(zoomed_tensor, (pad_width, pad_width + pad_width_extra, pad_height, pad_height + pad_height_extra), 'reflect', 0)
    else:
        # For zoom_factor > 1, center crop to original dimensions
        start_row = (zoomed_tensor.shape[2] - original_height) // 2
        start_col = (zoomed_tensor.shape[3] - original_width) // 2
        zoomed_tensor = zoomed_tensor[:, :, start_row:start_row + original_height, start_col:start_col + original_width]
    
    return zoomed_tensor.squeeze(0).permute(1,2,0)  # Remove batch dimension before returning

def ten2img(ten):
    return ten.cpu().numpy().astype(np.uint8)
import matplotlib.pyplot as plt
#%% Inits
cam = lt.WebCam(cam_id=-1, shape_hw=shape_cam)
cam.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)

# Diffusion Pipe
pipe = AutoPipelineForImage2Image.from_pretrained(model_turbo, torch_dtype=torch.float16, variant="fp16", local_files_only=True)
pipe.to("cuda")
pipe.vae = AutoencoderTiny.from_pretrained(model_vae, torch_device='cuda', torch_dtype=torch.float16, local_files_only=True)
pipe.vae = pipe.vae.cuda()
pipe.set_progress_bar_config(disable=True)
pipe.unet.forward = forward_modulated.__get__(pipe.unet, UNet2DConditionModel)


    
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
    
cam_img = cam.get_img()
device = torch.device("cuda")
    
# human body segmentation
human_seg = HumanSeg(model_name='deeplabv3_mobilenet_v3_large')

# predict depth
# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_depth_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

model_depth = torch.hub.load("intel-isl/MiDaS", model_depth_type)
model_depth.to(device)
model_depth.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_depth_type == "DPT_Large" or model_depth_type == "DPT_Hybrid":
    transform_depth = midas_transforms.dpt_transform
else:
    transform_depth = midas_transforms.small_transform

def get_depth(image):
    depth_predicted = model_depth(transform_depth(image).to(device))
    
    depth_map = torch.nn.functional.interpolate(
        depth_predicted.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    return depth_map




background_image = cv2.imread('seed_images/forest.png')
background_image = cv2.resize(background_image, (cam_img.shape[1], cam_img.shape[0]))

# Promptblender
blender = PromptBlender(pipe)

# Promptmanager
promptmanager = PromptManager(use_community_prompts)
prompt = promptmanager.get_new_prompt()

fract = 0
blender.set_prompt1(prompt)
blender.set_prompt2(prompt)
prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.blend_stored_embeddings(fract)
# prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender
blender.blend_stored_embeddings(fract)
# prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)

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

meta_input = lt.MetaInput()

memory_matrix = np.linspace(0.1,0.4,cam_img.shape[1])
memory_matrix = np.expand_dims(np.expand_dims(memory_matrix, 0), -1)
speech_detector = lt.Speech2Text()

# noise
latents = blender.get_latents()
noise_img2img_orig = torch.randn((1,4,noise_resolution_h,noise_resolution_w)).half().cuda()

image_displacement_accumulated = 0
image_displacement_array_accumulated = None

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

blur = MedianBlur((7, 7))

def get_noise_for_modulations(shape):
    return torch.randn(shape, device=pipe.device, generator=torch.Generator(device=pipe.device).manual_seed(1)).half()

modulations = {}
modulations_noise = {}
for i in range(3):
    modulations_noise[f'e{i}'] = get_noise_for_modulations(get_sample_shape_unet(f'e{i}'))
    modulations_noise[f'd{i}'] = get_noise_for_modulations(get_sample_shape_unet(f'd{i}'))
    
modulations_noise['b0'] = get_noise_for_modulations(get_sample_shape_unet('b0'))
    
prompt_decoder = 'fire'
prompt_embeds_decoder, negative_prompt_embeds_decoder, pooled_prompt_embeds_decoder, negative_pooled_prompt_embeds_decoder = blender.get_prompt_embeds(prompt_decoder, negative_prompt)

last_render_timestamp = time.time()
fract = 0
use_modulated_unet = True
while True:
    do_fix_seed = not meta_input.get(akai_midimix='F3', button_mode='toggle')
    if do_fix_seed:
        torch.manual_seed(0)
        
    noise_img2img_fresh = torch.randn((1,4,noise_resolution_h,noise_resolution_w)).half().cuda()#randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    
    noise_mixing = meta_input.get(akai_midimix="D0", val_min=0, val_max=1.0, val_default=0)
    noise_img2img = blender.interpolate_spherical(noise_img2img_orig, noise_img2img_fresh, noise_mixing)
    do_cam_coloring = meta_input.get(akai_midimix="G3", button_mode="toggle")
    do_gray_noise = meta_input.get(akai_midimix="G4", button_mode="toggle")
    do_record_mic = meta_input.get(akai_midimix="A3", akai_lpd8="A1", button_mode="held_down")
    
    if do_record_mic:
        if not speech_detector.audio_recorder.is_recording:
            speech_detector.start_recording()
    elif not do_record_mic:
        if speech_detector.audio_recorder.is_recording:
            try:
                prompt_prev = prompt
                prompt = speech_detector.stop_recording()
                print(f"New prompt: {prompt}")
                stop_recording = False
                fract = 0
                blender.set_prompt1(prompt_prev, negative_prompt)
                blender.set_prompt2(prompt, negative_prompt)
                
            except Exception as e:
                print(f"FAIL {e}")
            
    get_new_prompt = meta_input.get(akai_midimix='B3', akai_lpd8='A0', button_mode='pressed_once')
    if get_new_prompt:
        try:
            prompt_prev = prompt
            prompt = promptmanager.get_new_prompt()
            print(f"New prompt: {prompt}")
            stop_recording = False
            fract = 0
            blender.set_prompt1(prompt_prev, negative_prompt)
            blender.set_prompt2(prompt, negative_prompt)
        except Exception as e:
            print(f"fail! {e}")
            
    
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.blend_stored_embeddings(fract)

    save_prompt = meta_input.get(akai_midimix='B4', button_mode='pressed_once')
    if save_prompt:
        promptmanager.save_harvested_prompt(prompt)
    
    # # save_midi_settings = meta_input.get(akai_midimix='D4', button_mode='pressed_once')
    # # if save_midi_settings:
        
    #     path_midi_dump = "../submersion/midi_dumps"
    #     fn = None
    #     os.makedirs(path_midi_dump, exist_ok=True)
    #     parameters = []
    #     from datetime import datetime
    #     import yaml
    #     if fn == None:
    #         current_datetime_string = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    #         fn = f'midi_dump_{current_datetime_string}.yml'
    #     fp = os.path.join(path_midi_dump, fn)
    #     for id_, name in meta_input.id_name.items():
    #         value = meta_input.id_value[id_]
    #         parameters.append({'id':id_, 'name':name, 'value':value})
        
    #     parameters.append({'prompt':prompt})
    #     with open(fp, 'w') as file:
    #         yaml.dump(parameters, file)
        # akai_midimix.yaml_dump(path=path_midi_dump, prompt=prompt)
    
    cam_img = cam.get_img()
    cam_img = np.flip(cam_img, axis=1)
    
    # mask the body
    apply_body_mask = meta_input.get(akai_midimix="E3", button_mode="toggle")
    mask = human_seg.get_mask(cam_img)
    if apply_body_mask:
        cam_img *= np.expand_dims(mask, axis=2)
        cam_img += (1-np.expand_dims(mask, axis=2)) * background_image
    
    # test resolution
    cam_img = cv2.resize(cam_img.astype(np.uint8), (cam_resolution_w, cam_resolution_h))
    cam_img_raw = cam_img.copy()
    
    # do_aug_overlay = meta_input.get(akai_midimix='C3', button_mode='toggle')
    # if do_aug_overlay:
    #     aug_overlay = np.roll(aug_overlay,-10, axis=0)
    #     mask_aug = aug_overlay[:,:,0] != 0
    #     cam_img[mask_aug] = aug_overlay[mask_aug]
    
    strength = meta_input.get(akai_midimix="C1", val_min=0.5, val_max=1.0, val_default=0.5)
    num_inference_steps = 2 #int(meta_input.get(akai_midimix="C2", val_min=2, val_max=10, val_default=2))
    guidance_scale = meta_input.get(akai_midimix="C2", val_min=0, val_max=1, val_default=0.5)
    # guidance_scale = 1
    
    cam_img_torch = torch.from_numpy(cam_img.copy()).to(latents.device).float()
    
    cam_img_torch = blur(cam_img_torch.permute([2,0,1])[None])[0].permute([1,2,0])
    
    torch_last_diffusion_image = torch.from_numpy(last_diffusion_image).to(cam_img_torch)
    do_zoom = meta_input.get(akai_midimix="H4", akai_lpd8="C0", button_mode="toggle")
    if do_zoom:
        zoom_factor = meta_input.get(akai_midimix="F0", akai_lpd8="G0", val_min=0.8, val_max=1.2, val_default=1)
        torch_last_diffusion_image = zoom_image_torch(torch_last_diffusion_image, zoom_factor)
    if do_cam_coloring:
        for c in range(3):
            mask = (torch.rand(cam_img_torch.shape[0], cam_img_torch.shape[1], 1) < 0.8).repeat(1, 1, 3)
            mask[:,:,c] = 0
            cam_img_torch[mask] = 255


    if do_add_noise:
        # coef noise
        coef_noise = meta_input.get(akai_midimix="E0", val_min=0, val_max=0.3, val_default=0.03)
        
        if not do_gray_noise:
            t_rand_r = (torch.rand(cam_img_torch.shape[0], cam_img_torch.shape[1], 1, device=cam_img_torch.device) - 0.5) * coef_noise * 255 * 5
            t_rand_g = (torch.rand(cam_img_torch.shape[0], cam_img_torch.shape[1], 1, device=cam_img_torch.device) - 0.5) * coef_noise * 255 * 5
            t_rand_b = (torch.rand(cam_img_torch.shape[0], cam_img_torch.shape[1], 1, device=cam_img_torch.device) - 0.5) * coef_noise * 255 * 5
            
            t_rand_r[t_rand_r<0.5] = 0
            t_rand_g[t_rand_g<0.5] = 0
            t_rand_b[t_rand_b<0.5] = 0
            
            # Combine the independent noise for each channel
            t_rand = torch.cat((t_rand_r, t_rand_g, t_rand_b), dim=2)

        else:
            t_rand = (torch.rand(cam_img_torch.shape, device=cam_img_torch.device)[:,:,0].unsqueeze(2) - 0.5) * coef_noise * 255 * 5
        cam_img_torch += t_rand
        torch_last_diffusion_image += t_rand
        # cam_img_torch += (torch.rand(cam_img_torch.shape, device=cam_img_torch.device)[:,:,0].unsqueeze(2) - 0.5) * coef_noise * 255 * 5
    

    do_accumulate_acid = meta_input.get(akai_midimix="C4", akai_lpd8="B0", button_mode="toggle")
    do_local_accumulate_acid = meta_input.get(akai_midimix="D4", button_mode="toggle")
    invert_accumulate_acid = meta_input.get(akai_midimix="D3", akai_lpd8="B1", button_mode="toggle")
    # acid_persistence = meta_input.get(akai_midimix="D1", val_min=0.01, val_max=0.99, val_default=0.5)
    # acid_decay = meta_input.get(akai_midimix="D2", val_min=0.01, val_max=0.5, val_default=0.2)
    
    if do_accumulate_acid:
        ## displacement controlled acid
        if last_cam_img_torch is None:
            last_cam_img_torch = cam_img_torch
        acid_gain = meta_input.get(akai_midimix="C0", akai_lpd8="F0", val_min=0, val_max=1.0, val_default=0.05)
            
        image_displacement_array = ((cam_img_torch - last_cam_img_torch)/255)**2
        
        if do_local_accumulate_acid:
            image_displacement_array = (1-image_displacement_array*100)
            image_displacement_array = image_displacement_array.clamp(0)
            if image_displacement_array_accumulated == None:
                image_displacement_array_accumulated = torch.zeros_like(image_displacement_array)           
            image_displacement_array_accumulated[image_displacement_array>=0.5] += 2e-2
            image_displacement_array_accumulated[image_displacement_array<0.5] -= 2e-1
            image_displacement_array_accumulated = image_displacement_array_accumulated.clamp(0)
            
            image_displacement_array_accumulated = image_displacement_array_accumulated.mean(2, keepdims=True)
            image_displacement_array_accumulated = image_displacement_array_accumulated.repeat([1,1,3])
            
            image_displacement_array_accumulated -= image_displacement_array_accumulated.min()
            image_displacement_array_accumulated /= image_displacement_array_accumulated.max()
            
            if invert_accumulate_acid:
                acid_array = 1-image_displacement_array_accumulated
                acid_array[acid_array<0.05]=0.05
                acid_array *= acid_gain                
            else:
                acid_array = (image_displacement_array_accumulated)*acid_gain

        
        else:
            image_displacement = image_displacement_array.mean()
            image_displacement = (1-image_displacement*100)
            if image_displacement < 0:
                image_displacement = 0
                
            if image_displacement >= 0.5:
                image_displacement_accumulated += 2e-2
            else:
                image_displacement_accumulated -= 4e-1
            # if image_displacement >= 0.5:
            #     image_displacement_accumulated += acid_persistence
            # else:
            #     image_displacement_accumulated -= (1-acid_persistence)
                
            if image_displacement_accumulated < 0:
                image_displacement_accumulated = 0
            
            if invert_accumulate_acid:
                acid_strength = max(0.1, 1 - image_displacement_accumulated)
            else:
                acid_strength = 0.1 + image_displacement_accumulated * 1
            acid_strength *= acid_gain
            acid_strength = min(1, acid_strength)
        last_cam_img_torch = cam_img_torch.clone()
    else:
        acid_strength = meta_input.get(akai_midimix="C0", akai_lpd8="F0", val_min=0, val_max=0.8, val_default=0.11)
        
    F2 = meta_input.get(akai_midimix="F2", val_min=0, val_max=10.0, val_default=0)
    if F2 > 0:
        acid_strength = (np.sin(F2*float(time.time())) + 1)/2
        
    # just a test
    # cam_img_torch = (1.-acid_strength)*cam_img_torch + acid_strength*torch.from_numpy(last_diffusion_image).to(cam_img_torch)
    if do_accumulate_acid and do_local_accumulate_acid:
        cam_img_torch = (1.-acid_array)*cam_img_torch + acid_array*torch_last_diffusion_image
    else:
        cam_img_torch = (1.-acid_strength)*cam_img_torch + acid_strength*torch_last_diffusion_image
    # if meta_input.get(akai_midimix='E4', button_mode='pressed_once'):
    #     xxx
    cam_img_torch = torch.clamp(cam_img_torch, 0, 255)
    cam_img = cam_img_torch.cpu().numpy()
        
    if use_modulated_unet:
        H2 = meta_input.get(akai_midimix="H2", val_min=0, val_max=10, val_default=0)
        modulations['b0_samp'] = torch.tensor(H2, device=latents.device)
        modulations['e2_samp'] = torch.tensor(H2, device=latents.device)
        
        H1 = meta_input.get(akai_midimix="H1", val_min=1, val_max=10, val_default=2)
        modulations['b0_emb'] = torch.tensor(H1, device=latents.device)
        modulations['e2_emb'] = torch.tensor(H1, device=latents.device)
        
        fract_mod = meta_input.get(akai_midimix="G0", val_default=0, val_max=2, val_min=0)
        if fract_mod > 1:
            modulations['d*_extra_embeds'] = prompt_embeds_decoder    
        else:
            modulations['d*_extra_embeds'] = prompt_embeds
            
        modulations['modulations_noise'] = modulations_noise
        
    if use_modulated_unet:
        cross_attention_kwargs ={}
        cross_attention_kwargs['modulations'] = modulations
    else:
        cross_attention_kwargs = None
    
    use_debug_overlay = meta_input.get(akai_midimix="H3", akai_lpd8="D1", button_mode="toggle")
    if use_debug_overlay:
        image = cam_img.astype(np.uint8)
        if do_local_accumulate_acid:
            try:
                image = (image_displacement_array_accumulated*255).cpu().numpy().astype(np.uint8)
            except Exception as e:
                print(f"bad error! {e}")
    else:
        image = pipe(image=Image.fromarray(cam_img.astype(np.uint8)), 
                      latents=latents, num_inference_steps=num_inference_steps, strength=strength, 
                      guidance_scale=guidance_scale, prompt_embeds=prompt_embeds, 
                      negative_prompt_embeds=negative_prompt_embeds, 
                      pooled_prompt_embeds=pooled_prompt_embeds, 
                      negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, noise_img2img=noise_img2img, 
                      modulations=modulations,cross_attention_kwargs=cross_attention_kwargs).images[0]
        
        image_background = pipe(image=Image.fromarray(cv2.resize(background_image.astype(np.uint8), (cam_resolution_w, cam_resolution_h))), 
                      latents=latents, num_inference_steps=num_inference_steps, strength=strength, 
                      guidance_scale=guidance_scale, prompt_embeds=prompt_embeds, 
                      negative_prompt_embeds=negative_prompt_embeds, 
                      pooled_prompt_embeds=pooled_prompt_embeds, 
                      negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, noise_img2img=noise_img2img, 
                      modulations=modulations,cross_attention_kwargs=cross_attention_kwargs).images[0]
        
    time_difference = time.time() - last_render_timestamp
    last_render_timestamp = time.time()
    
    # lt.dynamic_print(f'fps: {np.round(1/time_difference)}')

    
    last_diffusion_image = np.array(image, dtype=np.float32)
    
    do_antishift = meta_input.get(akai_midimix="A4", akai_lpd8="D0", button_mode="toggle")
    x_shift = meta_input.get(akai_midimix="B0", akai_lpd8="H0", val_default=0, val_max=10, val_min=-10)
    y_shift = meta_input.get(akai_midimix="B1", akai_lpd8="H1", val_default=0, val_max=10, val_min=-10)
    if do_antishift:
        last_diffusion_image = np.roll(last_diffusion_image,int(y_shift),axis=0)
        last_diffusion_image = np.roll(last_diffusion_image,int(x_shift),axis=1)
        # last_diffusion_image = zoom_image(last_diffusion_image, 1.5)
    
    mask = cv2.resize(mask.astype(np.uint8), (cam_resolution_w, cam_resolution_h))    
    
    image_arr = np.array(image)
    # iamge_depth = get_depth(image_arr)
    # iamge_depth /= iamge_depth.max()
    # iamge_depth *= 30
    
    image_grayscale = np.mean(image_arr, 2) * mask
    image_background_grayscale = np.array(image_background).mean(2).astype(np.uint8)
    image_arr[:,:,0] = image_grayscale
    image_arr[:,:,0] += (1 - mask)*image_background_grayscale
    image_arr[:,:,1:] = 0
    
    right_eye = image_grayscale
    right_eye = np.roll(right_eye, -15, axis=1)
    mask_right_eye_background = right_eye == 0
    
    image_arr[:,:,2] = right_eye + mask_right_eye_background*image_background_grayscale
    
    # Render the image
    renderer.render(image_arr)
    
    # move fract forward
    d_fract_embed = meta_input.get(akai_midimix="A1", akai_lpd8="E0", val_min=0.0005, val_max=0.05, val_default=0.005)
    fract += d_fract_embed
    fract = np.clip(fract, 0, 1)
    

        
        
