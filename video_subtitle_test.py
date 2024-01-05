#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:32:57 2024

@author: lunar
"""

#%%
from PIL import Image, ImageDraw, ImageFont
import numpy as np


#%%
class SubtitleAdder:
    def __init__(self, font_path='arial.ttf', font_size=30, font_color=(255,255,0), line_spacing=10):
        self.font_path = font_path
        self.font_size = font_size
        self.font_color = font_color
        self.line_spacing = line_spacing

    def get_subtitle_position(self, img_res, offset_from_bottom):
        """
        Calculate the starting position for the first line of subtitles.

        Args:
        img_res (list): Image resolution [width, height].
        offset_from_bottom (int): Vertical offset from the bottom for the first line.

        Returns:
        tuple: Position (x, y) for the first subtitle line.
        """
        width, height = img_res
        x_position = width // 2
        y_position = height - offset_from_bottom
        return (x_position, y_position)

    def add_subtitles(self, image, text, img_res, offset_from_bottom=50):
        """
        Add multi-line subtitles to an image.

        Args:
        image (np.array): The image array.
        text (str): Subtitle text, possibly multi-line.
        img_res (list): Image resolution [width, height].
        offset_from_bottom (int): Offset from the bottom for the last subtitle line.

        Returns:
        np.array: The image with subtitles.
        """
        lines = text.split('\n')
        subtitle_position = self.get_subtitle_position(img_res, offset_from_bottom + len(lines) * self.font_size + (len(lines) - 1) * self.line_spacing)
            
        print(subtitle_position)
        pil_img = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype(self.font_path, self.font_size)

        for i,line in enumerate(lines[::-1]):  # Reverse the lines to start from the bottom
            width = draw.textlength(line, font=font)
            x, y = subtitle_position
            if i == 0:
                x -= width // 2  # Center align text
            print(x)
            draw.text((x, y), line, font=font, fill=self.font_color)
            subtitle_position = (x, y - self.font_size - self.line_spacing)  # Move up for the next line

        return np.array(pil_img)


#%%
# Example usage
img_res = [1024, 1024]
image = np.zeros((img_res[1], img_res[0], 3), dtype=np.uint8)  # Placeholder for an image

subtitle_adder = SubtitleAdder()
multi_line_text = "First line of subtitle\nSecond line of subtitle\nThird line of subtitle"
modified_image = subtitle_adder.add_subtitles(image, multi_line_text, img_res)
im = Image.fromarray(modified_image)
im.show()


