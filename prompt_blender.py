#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:30:17 2023

@author: lunar
"""

import torch

class PromptBlender:
    def __init__(self, pipe):
        self.pipe = pipe

    @staticmethod
    @torch.no_grad()
    def interpolate_spherical(p0, p1, fract_mixing: float):
        """
        Helper function to correctly mix two random variables using spherical interpolation.
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

    def get_prompt_embeds(self, prompt, negative_prompt=""):
        """
        Encodes a text prompt into embeddings using the model pipeline.
        """
        (
         prompt_embeds, 
         negative_prompt_embeds, 
         pooled_prompt_embeds, 
         negative_pooled_prompt_embeds
         ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=0,
            clip_skip=False,
        )
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def blend_prompts(self, embeds1, embeds2, fract):
        """
        Blends two sets of prompt embeddings based on a specified fraction.
        """
        prompt_embeds1, negative_prompt_embeds1, pooled_prompt_embeds1, negative_pooled_prompt_embeds1 = embeds1
        prompt_embeds2, negative_prompt_embeds2, pooled_prompt_embeds2, negative_pooled_prompt_embeds2 = embeds2

        blended_prompt_embeds = self.interpolate_spherical(prompt_embeds1, prompt_embeds2, fract)
        blended_negative_prompt_embeds = self.interpolate_spherical(negative_prompt_embeds1, negative_prompt_embeds2, fract)
        blended_pooled_prompt_embeds = self.interpolate_spherical(pooled_prompt_embeds1, pooled_prompt_embeds2, fract)
        blended_negative_pooled_prompt_embeds = self.interpolate_spherical(negative_pooled_prompt_embeds1, negative_pooled_prompt_embeds2, fract)

        return blended_prompt_embeds, blended_negative_prompt_embeds, blended_pooled_prompt_embeds, blended_negative_pooled_prompt_embeds

    def blend_sequence_prompts(self, prompts, n_steps):
        """
        Generates a sequence of blended prompt embeddings for a list of text prompts.
        """
        blended_prompts = []
        for i in range(len(prompts) - 1):
            prompt_embeds1 = self.get_prompt_embeds(prompts[i])
            prompt_embeds2 = self.get_prompt_embeds(prompts[i + 1])
            for step in range(n_steps):
                fract = step / float(n_steps - 1)
                blended = self.blend_prompts(prompt_embeds1, prompt_embeds2, fract)
                blended_prompts.append(blended)
        return blended_prompts


#%% FOR ASYNC INSERTION OF PROMPTS
    
class PromptBlenderAsync:
    def __init__(self, pipe, initial_prompts, n_steps):
        self.pipe = pipe
        self.prompts = initial_prompts
        self.n_steps = n_steps
        self.blended_prompts = self.blend_sequence_prompts(self.prompts, self.n_steps)
        
        
    def add_prompt(self, new_prompt):
        """
        Adds a new prompt to the sequence and updates the blended prompts.
        """
        self.prompts.append(new_prompt)
        self.blended_prompts = self.blend_sequence_prompts(self.prompts, self.n_steps)

    @staticmethod
    @torch.no_grad()
    def interpolate_spherical(p0, p1, fract_mixing: float):
        """
        Helper function to correctly mix two random variables using spherical interpolation.
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

    def get_prompt_embeds(self, prompt, negative_prompt=""):
        """
        Encodes a text prompt into embeddings using the model pipeline.
        """
        (
         prompt_embeds, 
         negative_prompt_embeds, 
         pooled_prompt_embeds, 
         negative_pooled_prompt_embeds
         ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=0,
            clip_skip=False,
        )
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def blend_prompts(self, embeds1, embeds2, fract):
        """
        Blends two sets of prompt embeddings based on a specified fraction.
        """
        prompt_embeds1, negative_prompt_embeds1, pooled_prompt_embeds1, negative_pooled_prompt_embeds1 = embeds1
        prompt_embeds2, negative_prompt_embeds2, pooled_prompt_embeds2, negative_pooled_prompt_embeds2 = embeds2

        blended_prompt_embeds = self.interpolate_spherical(prompt_embeds1, prompt_embeds2, fract)
        blended_negative_prompt_embeds = self.interpolate_spherical(negative_prompt_embeds1, negative_prompt_embeds2, fract)
        blended_pooled_prompt_embeds = self.interpolate_spherical(pooled_prompt_embeds1, pooled_prompt_embeds2, fract)
        blended_negative_pooled_prompt_embeds = self.interpolate_spherical(negative_pooled_prompt_embeds1, negative_pooled_prompt_embeds2, fract)

        return blended_prompt_embeds, blended_negative_prompt_embeds, blended_pooled_prompt_embeds, blended_negative_pooled_prompt_embeds

    def blend_sequence_prompts(self, prompts, n_steps):
        """
        Generates a sequence of blended prompt embeddings for a list of text prompts.
        """
        blended_prompts = []
        for i in range(len(prompts) - 1):
            prompt_embeds1 = self.get_prompt_embeds(prompts[i])
            prompt_embeds2 = self.get_prompt_embeds(prompts[i + 1])
            for step in range(n_steps):
                fract = step / float(n_steps - 1)
                blended = self.blend_prompts(prompt_embeds1, prompt_embeds2, fract)
                blended_prompts.append(blended)
        return blended_prompts
    
    
    
