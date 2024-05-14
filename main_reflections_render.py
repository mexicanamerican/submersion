#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
grab the image and run gpt4 vision. exclude prompts that are too close!
battle proof!
    - works offlnie
    - no akaki
    - try except handling everywhere (generic fail)
save the transformed images from each run
new seed every new run


end of experience?
get rid of midimix
prompt tuning?
try other prompts
cleanup & comments
try blurring cam image?
fine-tune the focus
"""

#%%
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import os
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

#%% DIFFUSION PARAMETERS
shape_cam= (1080, 1920)


# Cam & crop parameters
cam_focus = 300
autofocus = 0 # 0 is disabled, 1 enabled
precrop_shape_cam = (750, 750) # we always precrop with this shape
padding_face_crop = 150 # how much space around the face padded


size_diff_img = (512, 512) # size of diffusion gen. 512 ideal.
size_render = (1080, int(1080*precrop_shape_cam[1]/precrop_shape_cam[0]))  # render window size
human_mask_boundary_relative = 0.1
guidance_scale = 0.0 # leave


# %% INITS
cam = lt.WebCam(cam_id=0, shape_hw=shape_cam)
cam.cam.set(cv2.CAP_PROP_AUTOFOCUS, autofocus)
cam.cam.set(cv2.CAP_PROP_FOCUS, cam_focus)

pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", local_files_only=True)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)


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

model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
yolo = YOLO(model_path)

logger = lt.LogPrint()

#%%

def get_prompt():
    # return get_prompt_celebrity()
    # return get_prompt_emo()
    # return get_prompt_facial_features()
    # return get_prompt_nationalities_age_emotion()
    return get_prompt_nationalities_age_emotion()


def get_prompt_nationalities_age_emotion():
    list_nationalities = ["capo verdian", "nigerian", "moroccan", "turkish", "persian", "spanish", "american", "japanese", "chinese", "mongolian", "russian", "Brazilian", "Australian", "Egyptian", "Finnish", "Canadian", "Argentinian", "South Korean", "Kenyan", "Ukrainian", "Norwegian"]
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

def get_prompt_face_mouth():
    prompt = f"close-up portrait of a person with the mouth full open"
    return prompt

def get_prompt_emo():
    emotion = random.choice(['happy', 'sad'])
    prompt = f"close-up portrait of a {emotion} person"
    return prompt

def get_prompt_facial_features():
    attribute = random.choice(['tiny', 'huge'])
    feature = random.choice(['nose', 'eyes', 'mouth'])
    prompt = f"close-up portrait of a person with a {attribute} {feature}"
    return prompt



#%% IMPORTANT PARAMS - MOVE UP LATER
total_time_experience = 2*60 # in seconds
time_wait_camfeed = 0.2 # emulating low fps when we just have the cam feedthrough :)
nmb_face_detection_streak_required = 5 # switching on the experience (new person)
nmb_no_face_detection_streak_required = 15 #switching off the experience (person left)
num_inference_steps_max = 50 # maximum
do_automatic_experience_progression = True # Required for automatic increase of effect during experience
num_inference_steps_min_start = 15 # the value at the beginning of exp
num_inference_steps_min_end = 5 # the value at end of experience
dt_transform_in = 12 # buildup time (linear)
dt_transform_stay = 6 # how long to stay at current max transform
dt_transform_out = 5 # return time to camera feed
do_auto_face_y_adjust = True
do_save_images = True # saves the imags at maximum (for debug)

# AUTO PARAMS
human_mask_boundary = int(human_mask_boundary_relative*min(size_diff_img))
is_transformation_active = False
list_scores = []
prompt = "photo of a very young person"
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

nmb_face_detection_current = 0
nmb_no_face_detection_current = 0

# time based setup
fract_experience = 0 # auto
num_inference_steps_min = 10 # auto
yshift = 0 #auto
current_seed = 420
save_current_image = False

while True:
    t0 = time.time()
    # First check if there is a face present
    cam_img = cam.get_img()
    cam_img = np.flip(cam_img, axis=1)
    cam_img = np.uint8(cam_img)
    
    if (cam_img.shape[0] - precrop_shape_cam[0]) // 2 - yshift < 0 or (cam_img.shape[0] + precrop_shape_cam[0]) // 2 - yshift >= cam_img.shape[0]:
        yshift = 0
    
    cam_img = cam_img[(cam_img.shape[0] - precrop_shape_cam[0]) // 2 - yshift:(cam_img.shape[0] + precrop_shape_cam[0]) // 2 -yshift, (cam_img.shape[1] - precrop_shape_cam[1]) // 2:(cam_img.shape[1] + precrop_shape_cam[1]) // 2]
    
    cropping_coordinates = crop_face_square(cam_img, padding=padding_face_crop)
    if cropping_coordinates is None:
        is_face_present_current_frame = False
    else:
        is_face_present_current_frame = True
        
    # dt_transform_in = meta_input.get(akai_midimix="A0", val_min=1, val_max=20)
    # dt_transform_stay = meta_input.get(akai_midimix="A1", val_min=1, val_max=20)
    # dt_transform_out = meta_input.get(akai_midimix="A2", val_min=1, val_max=20)
    
    if not do_automatic_experience_progression:
        num_inference_steps_min = meta_input.get(akai_midimix="B0", val_min=2, val_max=5, val_default=4)
        num_inference_steps_max = meta_input.get(akai_midimix="B1", val_min=6, val_max=50, val_default=50)
    
    
    p_start_transform = meta_input.get(akai_midimix="B2", val_min=0, val_max=0.1)
    
    manual_num_inference_overrider = meta_input.get(akai_midimix='C4', button_mode='toggle')
    
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
    
    # ACTIVATE EXPERIENCE? use the face counters
    if not is_experience_active and nmb_face_detection_current >= nmb_face_detection_streak_required:
        logger.print(f"Starting experience! Detected face for {nmb_face_detection_current} consecutive frames!", color="green")
        is_experience_active = True
        current_seed = np.random.randint(999999999999999)
        time_experience_started = time.time()
        idx_img = 0 # for saving
        if do_save_images:
            save_current_image = True
        if do_auto_face_y_adjust:
            y1 = cropping_coordinates[1]
            y2 = cropping_coordinates[3]
            ymean = 0.5 * (y1 + y2)
            yshift = int(precrop_shape_cam[0]/2 - ymean)
            yshift = np.clip(yshift, -cam_img.shape[0] //2 +1, cam_img.shape[0] //2 -1 )
            logger.print(f"Applying camera yshift to center face: {yshift}")
            

    # DEACTIVATE EXPERIENCE? use the face counters
    if is_experience_active and nmb_no_face_detection_current >= nmb_no_face_detection_streak_required:
        is_experience_active = False
        time_experience_stopped = time.time()
        logger.print(f"Stopping experience! Detected NO face for {nmb_no_face_detection_current} consecutive frames!", color="green")
        yshift = 0
        
    
    # Cycle the face detection already here, because of continue statement below
    is_face_present_previous_frame = is_face_present_current_frame
    # Was there a face present as well in the previous frame? If so, then 
    if not is_experience_active or not is_face_present_current_frame:
        # logger.print("waiting...")
        renderer.render(Image.fromarray(cam_img))
        time.sleep(time_wait_camfeed)
        is_active_transform = False
        continue
    
    
    cam_img_cropped = Image.fromarray(cam_img).crop(cropping_coordinates)
    size_cam_img_cropped_orig = cam_img_cropped.size
    cam_img_cropped = cam_img_cropped.resize(size_diff_img)
    

    if not is_active_transform and not manual_num_inference_overrider and np.random.rand() < p_start_transform:
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
        logger.print(f"New transform upcoming, start_transform=True. dt_transform_in {dt_transform_in} dt_transform_stay {dt_transform_stay} dt_transform_out {dt_transform_out}")
        
    if get_new_prompt:
        prompt = get_prompt()
        logger.print(f"new prompt: {prompt}", color="blue")
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)
        get_new_prompt = False
        
    if is_active_transform:
        dt_transform = time.time() - t_transform_started
        
        # decide which phase we are in
        if dt_transform >= dt_transform_in + dt_transform_stay + dt_transform_out:
            is_active_transform = False
            logger.print("Entering new phase: Staying with Camera feed")
            num_inference_steps = num_inference_steps_max
        elif dt_transform >= dt_transform_in + dt_transform_stay:
            fract_transform = (dt_transform - dt_transform_in - dt_transform_stay) / dt_transform_out
            num_inference_steps = fract_transform*(num_inference_steps_max-num_inference_steps_min) + num_inference_steps_min
            # logger.print(f"phase: transform out: fract_transform {fract_transform} num_inference_steps {num_inference_steps}")
            if not did_print_out:
                logger.print("Entering new phase: Max transform -> Camera feed")
                did_print_out = True
        elif dt_transform >= dt_transform_in:
            fract_transform = (dt_transform - dt_transform_in) / dt_transform_stay
            num_inference_steps = num_inference_steps_min
            if not did_print_stay:
                logger.print("Entering new phase: Staying with Max transform")
                did_print_stay = True
                if do_save_images:
                    save_current_image = True
                    idx_img += 1
            # logger.print(f"phase: transform stay: fract_transform {fract_transform} num_inference_steps {num_inference_steps}")
        else:
            fract_transform = dt_transform / dt_transform_in
            num_inference_steps = (1-fract_transform)*(num_inference_steps_max-num_inference_steps_min) + num_inference_steps_min
            # logger.print(f"phase: transform in: fract_transform {fract_transform} num_inference_steps {num_inference_steps}")
            if not did_print_in:
                logger.print("Entering new phase: Camera feed -> Max transform")                
                did_print_in = True
                if do_automatic_experience_progression:
                    fract_experience = (time.time() - time_experience_started) / total_time_experience
                    fract_experience = np.clip(fract_experience, 0, 1)
                    num_inference_steps_min = fract_experience * (num_inference_steps_min_end - num_inference_steps_min_start) + num_inference_steps_min_start
                    num_inference_steps_min = int(np.round(num_inference_steps_min))
                    logger.print(f"Automatic do_time_based num_inference: fract_experience {fract_experience:2.2f} num_inference_steps_min {num_inference_steps_min}")
                    
        
        num_inference_steps = np.clip(num_inference_steps, num_inference_steps_min, num_inference_steps_max)
        num_inference_steps = int(np.round(num_inference_steps))
        
    manual_num_inference_steps = meta_input.get(akai_midimix='C5', val_min=num_inference_steps_min, val_max=num_inference_steps_max)
    
    if manual_num_inference_overrider:
        num_inference_steps =int(manual_num_inference_steps)
    
        
    
    psychophysics_detection = meta_input.get(akai_midimix="H4", button_mode="pressed_once")
    if psychophysics_detection:
        logger.print(f"captured at fract_transform {fract_transform}")
        list_scores.append(fract_transform)
    
    
    torch.manual_seed(current_seed)
    
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
        
    # print(num_inference_steps)

    show_cam = meta_input.get(akai_midimix="H3", akai_lpd8="D1", button_mode="toggle")
    
    if show_cam:
        image_show = cam_img.astype(np.uint8)
    else:
        # Re-insert image_diffusion into cam_img at the cropped position
        try:
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

            if save_current_image:
                ar_imgs_folder = "ar_imgs"
                if not os.path.exists(ar_imgs_folder):
                    os.makedirs(ar_imgs_folder)
                timestamp = time.strftime("%y%m%d_%H%M", time.localtime(time_experience_started))
                filename = f"{timestamp}_{idx_img}.png"
                cam_img_pil.save(os.path.join(ar_imgs_folder, filename), format='JPEG', quality=90)
                idx_img += 1

                save_current_image = False

        except Exception as e:
            logger.print(f"inserting diffusion image fail: {e}")
        image_show = cam_img_pil
        
    renderer.render(image_show)
    
    
    
    
    


