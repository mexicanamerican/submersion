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


# pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
# pipe = pipe.to("cuda")

# pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
# pipe.vae = pipe.vae.cuda()

# pipe.set_progress_bar_config(disable=True)

# pipe = compile(pipe, config)


#%%

# Example usage
prompts = ["a man walking through the forest", "a man walking through the desert", "a man walking through the village", "a man walking through the war in the village","a man walking through the war in the village with explosions","a man walking through the destructed village, dead bodies, gore" , "a man walking through the desert", "a man walking through the forest"]
n_steps = 100
# blended_prompts = blend_sequence_prompts(prompts, n_steps)



#%%
sz = (512, 512)
# image = pipe(prompt="photo of a man", guidance_scale=0.0, num_inference_steps=4).images[0]


#%% 

from diffusers.utils import load_image
import cv2
from PIL import Image
import numpy as np


image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)
image
image = image.resize(512, 512)

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

#%%
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL

from diffusers.utils import load_image

import numpy as np

import torch

import cv2

from PIL import Image

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"

negative_prompt = "low quality, bad quality, sketches"

# download an image

image = load_image(

    "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"

)

# initialize the models and pipeline

controlnet_conditioning_scale = 0.0  # recommended for good generalization

controlnet = ControlNetModel.from_pretrained(

    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
    )


#%%
# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
# import torch
# controlnet = ControlNetModel.from_pretrained("lllyasviel/diffusers_xl_depth_full", torch_dtype=torch.float16)
#%%
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16")
# xxx
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "stabilityai/sdxl-turbo", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16"
# )
pipe = pipe.to("cuda")


#%%
controlnet_conditioning_scale = 0.5
t0 = time.time()
image = pipe("photo of a man", image=canny_image, controlnet_conditioning_scale=controlnet_conditioning_scale, guidance_scale=0.0, num_inference_steps=4).images[0]
dt = time.time() - t0
print(f"took {dt}s")