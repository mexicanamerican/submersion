#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch
import time
from diffusers import AutoencoderTiny
import xformers
import triton
import lunar_tools as lt
from PIL import Image
import numpy as np
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
import cv2
import sys
sys.path.append("../psychoactive_surface")

from prompt_blender import PromptBlender
from typing import List, Tuple
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters.kernels import get_binary_kernel2d
from img_utils import pad_image_to_width, pad_image_to_width, blend_images, process_cam_img, stitch_images, weighted_average_images
from datetime import datetime
from human_seg import HumanSeg
torch.set_grad_enabled(False)
from img_utils import center_crop_and_resize
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import random

model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
yolo = YOLO(model_path)
#%% PARAMETERS
compile_with_sfast = False
shape_cam= (1080, 1920)
# shape_cam= (720, 1280)
precrop_shape_cam = (750, 750)
size_diff_img = (512, 512)
size_render = (1080, int(1080*precrop_shape_cam[1]/precrop_shape_cam[0])) 
padding = 120
human_mask_boundary_relative = 0.1
guidance_scale = 0.0
# %% INITS
cam = lt.WebCam(cam_id=0, shape_hw=shape_cam)
cam.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)

pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", local_files_only=True)
pipe = pipe.to("cuda")
# pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
# pipe.vae = pipe.vae.cuda()
pipe.set_progress_bar_config(disable=True)

# if compile_with_sfast:
#     pipe.enable_xformers_memory_efficient_attention()
#     config = CompilationConfig.Default()
#     config.enable_xformers = True
#     config.enable_triton = True
#     config.enable_cuda_graph = True
#     pipe = compile(pipe, config)
    
    
# if compile_with_sfast:
#     from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)
#     config = CompilationConfig.Default()
#     config.enable_xformers = True
#     config.enable_triton = True
#     config.enable_cuda_graph = True
#     config.enable_jit = True
#     config.enable_jit_freeze = True
#     config.trace_scheduler = True
#     config.enable_cnn_optimization = True
#     config.preserve_parameters = False
#     config.prefer_lowp_gemm = True
#     pipe = compile(pipe, config)

# %% aux
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


def crop_face_square(cam_img, padding=30):
    try:
        # run face detection
        face_results = yolo(cam_img, verbose=False)
    #    If no face present, no need to do anything
        if len(face_results)==0:
            renderer.render(cam_img)
            return None
        
        # Initialize variables to store the maximum area and corresponding index
        max_area = 0
        index_of_largest_face = -1
        cropping_coordinates = None
        
        # Loop through all detected faces
        for i, result in enumerate(face_results):
            # Retrieve the bounding box coordinates in the format (x1, y1, x2, y2)
            x1, y1, x2, y2 = result.boxes.xyxy[0].cpu().numpy()
            # Calculate the area of the bounding box
            area = (x2 - x1) * (y2 - y1)
            
            # Check if this area is the largest we've seen so far
            if area > max_area:
                max_area = area
                index_of_largest_face = i
                # Update the cropping coordinates to the largest face found
                
                
        # Adjust to square
        
        # Calculate width and height of the detected face
        width = x2 - x1
        height = y2 - y1
        
        # Determine the amount to extend to make the bounding box square
        if width > height:
            difference = (width - height) // 2
            y1 = max(0, y1 - difference)  # Extend upwards
            y2 = min(cam_img.shape[0], y2 + difference)  # Extend downwards
        elif height > width:
            difference = (height - width) // 2
            x1 = max(0, x1 - difference)  # Extend to the left
            x2 = min(cam_img.shape[1], x2 + difference)  # Extend to the right
        
        # Adjust dimensions to make the bounding box square
        if (x2 - x1) != (y2 - y1):  # Check if not square due to clamping at image borders
            new_size = min(x2 - x1, y2 - y1)
            x2 = x1 + new_size
            y2 = y1 + new_size
        
        # Calculate new padding to center the padding around the box
        padding_x = (padding // 2, padding // 2)
        padding_y = (padding // 2, padding // 2)
        
        # Adjust padding if it causes the box to go out of image bounds
        if x1 - padding_x[0] < 0:
            padding_x = (x1, padding - x1)
        if x2 + padding_x[1] > cam_img.shape[1]:
            padding_x = (padding - (cam_img.shape[1] - x2), cam_img.shape[1] - x2)
        
        if y1 - padding_y[0] < 0:
            padding_y = (y1, padding - y1)
        if y2 + padding_y[1] > cam_img.shape[0]:
            padding_y = (padding - (cam_img.shape[0] - y2), cam_img.shape[0] - y2)
        
        # Apply adjusted padding
        x1 = max(0, x1 - padding_x[0])
        x2 = min(cam_img.shape[1], x2 + padding_x[1])
        y1 = max(0, y1 - padding_y[0])
        y2 = min(cam_img.shape[0], y2 + padding_y[1])
        
        cropping_coordinates = (int(x1), int(y1), int(x2), int(y2))
    
        return cropping_coordinates
    
    except Exception as e:
        return None
    

#%%
blender = PromptBlender(pipe)
human_seg = HumanSeg()
negative_prompt = 'blurry, tiled, wrong, bad art, pixels, amateur drawing, haze'
    
image_diffusion = Image.fromarray(np.zeros((size_diff_img[1], size_diff_img[0], 3), dtype=np.uint8))



latents = blender.get_latents()

renderer = lt.Renderer(width=size_render[1], height=size_render[0])
speech_detector = lt.Speech2Text()
meta_input = lt.MetaInput()

#%%

def get_prompt():
    # return get_prompt_celebrity()
    # return get_prompt_emo()
    return get_prompt_nationalities_age_emotion()


def get_prompt_nationalities_age_emotion():
    list_nationalities = ["nigerian", "moroccan", "turkish", "persian", "spanish", "american", "japanese", "chinese", "mongolian", "russian", "Brazilian", "Australian", "Egyptian", "Finnish", "Canadian", "Argentinian", "South Korean", "Kenyan", "Ukrainian", "Norwegian"]
    list_emotions = ["angry", "sad", "surprised", "happy", "smiling", "furious", "Ecstatic", "Melancholic", "Bewildered", "Content", "Nostalgic", "Anxious"]
    list_ages = ["middle aged", "very old", "very young", "teenager", "Toddler", "Elderly", "Young adult", "In their thirties"]
    
    nationality = random.choice(list_nationalities)
    emotion = random.choice(list_emotions)
    age = random.choice(list_ages)
    gender = random.choice(['male', 'female'])
    prompt = f"photo of a {emotion} and {age} {nationality} {gender} person"
    return prompt


def get_prompt_celebrity():
    list_celebs = [
        'donald trump', 'elon musk', 'Oprah Winfrey', 'Tom Cruise', 'Taylor Swift', 'Leonardo DiCaprio',
        'Beyoncé', 'Brad Pitt', 'Emma Watson', 'Cristiano Ronaldo', 'Angelina Jolie', 'Roger Federer',
        'Kim Kardashian', 'Justin Bieber', 'Serena Williams', 'Keanu Reeves', 'Ariana Grande', 'Will Smith',
        'Rihanna', 'Johnny Depp', 'Jennifer Lawrence', 'Lionel Messi', 'Scarlett Johansson', 'Shakira',
        'Dwayne Johnson', 'Lady Gaga', 'Robert Downey Jr.', 'Katy Perry', 'LeBron James', 'Adele'
    ]

    prompt = f"photo of {random.choice(list_celebs)}"
    return prompt

def get_prompt_emo():
    emotion = random.choice(['happy', 'sad'])
    prompt = f"close-up portrait of a {emotion} person"
    return prompt



#%%
human_mask_boundary = int(human_mask_boundary_relative*min(size_diff_img))
is_transformation_active = False
list_scores = []

prompt = "photo of a very old and very angry american person"
prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)

start_transform = False
t_transform_started = time.time()
is_active_transform = False
get_new_prompt = False
num_inference_steps = 30
t_transform_started = 0

is_experience_active = False
time_experience_started = 0
time_experience_stopped = 0
is_face_present_current_frame = False
is_face_present_previous_frame = False
nmb_face_detection_streak_required = 5
nmb_no_face_detection_streak_required = 5

nmb_face_detection_current = 0
nmb_no_face_detection_current = 0


while True:
    t0 = time.time()
        
    dt_transform_in = meta_input.get(akai_midimix="A0", val_min=1, val_max=20)
    dt_transform_stay = meta_input.get(akai_midimix="A1", val_min=1, val_max=20)
    dt_transform_out = meta_input.get(akai_midimix="A2", val_min=1, val_max=20)
    
    num_inference_steps_min = meta_input.get(akai_midimix="B0", val_min=2, val_max=5, val_default=4)
    num_inference_steps_max = meta_input.get(akai_midimix="B1", val_min=6, val_max=50, val_default=50)
    
    p_start_transform = meta_input.get(akai_midimix="B2", val_min=0, val_max=0.1)
    
    manual_strength_overrider = meta_input.get(akai_midimix='C4', button_mode='toggle')
    
    # First check if there is a face present
    cam_img = cam.get_img()
    cam_img = np.flip(cam_img, axis=1)
    cam_img = np.uint8(cam_img)
    cam_img = cam_img[(cam_img.shape[0] - precrop_shape_cam[0]) // 2:(cam_img.shape[0] + precrop_shape_cam[0]) // 2, (cam_img.shape[1] - precrop_shape_cam[1]) // 2:(cam_img.shape[1] + precrop_shape_cam[1]) // 2]
    
    cropping_coordinates = crop_face_square(cam_img, padding=150)
    if cropping_coordinates is None:
        is_face_present_current_frame = False
    else:
        is_face_present_current_frame = True
        
        
    # Count the number of subsequent frames where there was a face present
    if is_face_present_current_frame and is_face_present_previous_frame:
        nmb_face_detection_current += 1
    
    # If this is the first frame where we see a face, reset the counter
    if is_face_present_current_frame and not is_face_present_previous_frame:
        nmb_face_detection_current = 0 
        nmb_no_face_detection_current = 0 
        
    # Count the number of subsequent frames where there was NO face present
    if not is_face_present_current_frame and not is_face_present_previous_frame:
        nmb_no_face_detection_current += 1
    
    # If this is the first frame where we fail to see a face, reset the counter
    if not is_face_present_current_frame and is_face_present_previous_frame:
        nmb_no_face_detection_current = 0 
        nmb_face_detection_current = 0 
    
    # print(f"nmb_face_detection_current {nmb_face_detection_current} nmb_no_face_detection_current {nmb_no_face_detection_current}")
    
    # use the counters to activate or deactivate the experience
    if not is_experience_active and nmb_face_detection_current >= nmb_face_detection_streak_required:
        print(f"Starting experience! Detected face for {nmb_face_detection_current} consecutive frames!")
        is_experience_active = True
        time_experience_started = time.time()
        
    if is_experience_active and nmb_no_face_detection_current >= nmb_no_face_detection_streak_required:
        is_experience_active = False
        time_experience_stopped = time.time()
        print(f"Stopping experience! Detected NO face for {nmb_no_face_detection_current} consecutive frames!")
        
    
    # Cycle the face detection already here, because of continue statement
    is_face_present_previous_frame = is_face_present_current_frame
    # Was there a face present as well in the previous frame? If so, then 
    if not is_experience_active:
        # print("waiting...")
        renderer.render(Image.fromarray(cam_img))
        time.sleep(0.1)
        continue
    
    cam_img_cropped = Image.fromarray(cam_img).crop(cropping_coordinates)
    size_cam_img_cropped_orig = cam_img_cropped.size
    cam_img_cropped = cam_img_cropped.resize(size_diff_img)
    

    if not is_active_transform and not manual_strength_overrider and np.random.rand() < p_start_transform:
        start_transform = True
        
    start_transform_manually = meta_input.get(akai_midimix='C3', akai_lpd8='A0', button_mode='pressed_once')
    if start_transform_manually:
        start_transform = True
        
    if start_transform:
        t_transform_started = time.time()
        get_new_prompt = True
        start_transform = False
        is_active_transform = True
        did_print_in = False
        did_print_stay = False
        did_print_out = False
        print(f"start_transform True! {dt_transform_in} {dt_transform_stay} {dt_transform_out}")
        
    if get_new_prompt:
        prompt = get_prompt()
        print(f"new prompt: {prompt}")
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)
        get_new_prompt = False
        
    if is_active_transform:
        dt_transform = time.time() - t_transform_started
        
        # decide which phase we are in
        if dt_transform >= dt_transform_in + dt_transform_stay + dt_transform_out:
            is_active_transform = False
            print(f"phase: transform ended, back to normal")
            num_inference_steps = num_inference_steps_max
        elif dt_transform >= dt_transform_in + dt_transform_stay:
            fract_transform = (dt_transform - dt_transform_in - dt_transform_stay) / dt_transform_out
            num_inference_steps = fract_transform*(num_inference_steps_max-num_inference_steps_min) + num_inference_steps_min
            # print(f"phase: transform out: fract_transform {fract_transform} num_inference_steps {num_inference_steps}")
            if not did_print_out:
                print("phase: starting transform back")
                did_print_out = True
        elif dt_transform >= dt_transform_in:
            fract_transform = (dt_transform - dt_transform_in) / dt_transform_stay
            num_inference_steps = num_inference_steps_min
            if not did_print_stay:
                print("phase: starting staying transformed")
                did_print_stay = True
            # print(f"phase: transform stay: fract_transform {fract_transform} num_inference_steps {num_inference_steps}")
        else:
            fract_transform = dt_transform / dt_transform_in
            num_inference_steps = (1-fract_transform)*(num_inference_steps_max-num_inference_steps_min) + num_inference_steps_min
            # print(f"phase: transform in: fract_transform {fract_transform} num_inference_steps {num_inference_steps}")
            if not did_print_in:
                print("phase: starting transform")
                did_print_in = True
        
        num_inference_steps = np.clip(num_inference_steps, num_inference_steps_min, num_inference_steps_max)
        num_inference_steps = int(np.round(num_inference_steps))
        
    manual_num_inference_steps = meta_input.get(akai_midimix='C5', val_min=num_inference_steps_min, val_max=num_inference_steps_max)
    
    if manual_strength_overrider:
        num_inference_steps =int(manual_num_inference_steps)
        
    
    psychophysics_detection = meta_input.get(akai_midimix="H4", button_mode="pressed_once")
    if psychophysics_detection:
        print(f"captured at fract_transform {fract_transform}")
        list_scores.append(fract_transform)
    
    
    torch.manual_seed(420)
    
    # num_inference_steps = int(meta_input.get(akai_midimix="B2", akai_lpd8="H1", val_min=3, val_max=30, val_default=30))

    # only apply human mask in cam_img_cropped  
    human_mask = human_seg.get_mask(np.asarray(cam_img_cropped))

    strength = 1/num_inference_steps + 0.0001
    
    acid_strength = meta_input.get(akai_midimix="C0", val_min=0, val_max=0.8, val_default=0.11)
    input_image_diffusion = cam_img_cropped
    # input_image_diffusion = blend_images(image_diffusion, cam_img_cropped, acid_strength)
    
    image_diffusion = pipe(image=input_image_diffusion, 
                    latents=latents, num_inference_steps=num_inference_steps, strength=strength, 
                    guidance_scale=guidance_scale, prompt_embeds=prompt_embeds, 
                    negative_prompt_embeds=negative_prompt_embeds, 
                    pooled_prompt_embeds=pooled_prompt_embeds,          
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]
    
    
    dt = time.time() - t0
    fps = 1/dt
    
    show_fps = meta_input.get(akai_midimix="G3", button_mode="toggle")
    if show_fps:
        print(fps)

    show_cam = meta_input.get(akai_midimix="H3", akai_lpd8="D1", button_mode="toggle")
    
    if show_cam:
        image_show = cam_img.astype(np.uint8)
    else:
        # just show image_diffusion_mixed for now
        # Re-insert image_diffusion into cam_img at the cropped position
        human_mask = human_mask.astype(np.float32)
        human_mask[-human_mask_boundary:,:] = 0
        human_mask[:,-human_mask_boundary:] = 0
        human_mask[:,:human_mask_boundary] = 0
        human_mask[:human_mask_boundary,:] = 0
        human_mask = cv2.GaussianBlur(human_mask, (55, 55), 0)
        if human_mask.max() > 0:
            human_mask = human_mask/human_mask.max()
        image_diffusion_mixed = blend_images(np.array(image_diffusion), cam_img_cropped, human_mask)
        
        image_diffusion_mixed_resized = image_diffusion_mixed.resize(size_cam_img_cropped_orig)
        cam_img_pil = Image.fromarray(cam_img)
        cam_img_pil.paste(image_diffusion_mixed_resized, cropping_coordinates[:2])
        image_show = cam_img_pil
        
    renderer.render(image_show)
    
    
    
    
    


