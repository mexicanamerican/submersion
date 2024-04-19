#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import torch
import time
import lunar_tools as lt
import cv2

def stitch_images(img1, img2):
    # Determine the size for the new image
    width1, height1 = img1.size
    width2, height2 = img2.size
    new_width = width1 + width2
    new_height = max(height1, height2)

    # Create a new image with appropriate size
    new_img = Image.new('RGB', (new_width, new_height))

    # Paste the images side by side
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (width1, 0))

    return new_img


def pad_image_to_width(image, width_renderer):
    """
    Pads the given PIL image on the left and right with zeros to reach a specified width.

    Args:
    image (PIL.Image): The image to be padded.
    width_renderer (int): The final width of the image after padding.

    Returns:
    PIL.Image: The padded image.
    """
    current_width, _ = image.size
    total_padding = width_renderer - current_width
    if total_padding < 0:
        raise ValueError("width_renderer must be greater than the current image width.")

    # Divide the padding equally on both sides
    pad_width = total_padding // 2

    img_array = np.array(image)

    padded_array = np.pad(img_array, ((0, 0), (pad_width, pad_width), (0, 0)), mode='constant')

    # Convert the padded array back to a PIL image
    return Image.fromarray(padded_array)


def center_crop_and_resize(img, size=(512, 512)):
    """
    Center crop an image to the specified size and resize it.

    Args:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the cropped and resized image.
        size (tuple): Target size in the format (width, height). Default is (512, 512).
    """
    try:
        # Get the original dimensions of the image
        width, height = img.size

        # Calculate the coordinates for the center crop
        left = (width - size[0]) / 2
        top = (height - size[1]) / 2
        right = (width + size[0]) / 2
        bottom = (height + size[1]) / 2

        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))

        # Resize the cropped image to the specified size
        cropped_img = cropped_img.resize(size)

        return cropped_img
    except Exception as e:
        print(f"An error occurred: {e}")


def process_cam_img(cam_img, size=(512, 512)):
    cam_img = np.flip(cam_img, axis=1)
    cam_img = Image.fromarray(np.uint8(cam_img))
    cam_img = center_crop_and_resize(cam_img, size)
    return cam_img
    

def blend_images(img1, img2, weight):
    # Convert images to numpy arrays
    arr1 = np.array(img1, dtype=np.float64)
    arr2 = np.array(img2, dtype=np.float64)

    if isinstance(weight, Image.Image):
        weight = np.array(weight)

   
    if isinstance(weight, np.ndarray):
        assert np.max(weight) <= 1.0, "Maximum weight has to be 1.0"
        assert np.min(weight) >= 0.0, "Minimum weight has to be 0.0"
        # Ensure the mask is broadcastable to the image arrays
        weight = np.expand_dims(weight, axis=-1)
        # Ensure the weight array has the same number of channels as the images
        weight = np.repeat(weight, arr1.shape[-1], axis=-1)
        blended_arr = weight * arr1 + (1 - weight) * arr2
    else:
        assert 0 <= weight <= 1, "Weight must be between 0 and 1 inclusive."
        # Blend images with a single weight value
        blended_arr = weight * arr1 + (1 - weight) * arr2

    blended_arr = np.clip(blended_arr, 0, 255)

    # Convert back to image
    return Image.fromarray(blended_arr.astype(np.uint8))


def weighted_average_images(img_buffer, weight_begin, weight_end):
    num_images = len(img_buffer)
    
    # Generate linear weights
    weights = np.linspace(weight_begin, weight_end, num_images)

    # Convert all PIL images to numpy arrays and apply weights
    weighted_arrays = [np.array(img) * weight for img, weight in zip(img_buffer, weights)]

    # Sum and normalize the weighted arrays
    sum_weighted_arrays = np.sum(weighted_arrays, axis=0)
    avg_array = (sum_weighted_arrays / np.sum(weights)).astype(np.uint8)

    # Convert the average array back to a PIL image
    avg_image = Image.fromarray(avg_array)

    return avg_image

