import time
import torch

from diffusers import AutoPipelineForText2Image
from diffusers import (StableDiffusionPipeline,
                       EulerAncestralDiscreteScheduler)
from sfast.compilers.diffusion_pipeline_compiler import (compile,
                                                         CompilationConfig)

def load_model():
    # model = StableDiffusionPipeline.from_pretrained(
    #     'runwayml/stable-diffusion-v1-5',
    #     torch_dtype=torch.float16)
    
    model = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")

    model.scheduler = EulerAncestralDiscreteScheduler.from_config(
        model.scheduler.config)
    model.safety_checker = None
    model.to(torch.device('cuda:0'))
    return model

model = load_model()

config = CompilationConfig.Default()
# xformers and Triton are suggested for achieving best performance.
import xformers
config.enable_xformers = True
import triton
config.enable_triton = True
# CUDA Graph is suggested for small batch sizes and small resolutions to reduce CPU overhead.
# But it can increase the amount of GPU memory used.
# For StableVideoDiffusionPipeline it is not needed.
config.enable_cuda_graph = True

model = compile(model, config)

kwarg_inputs = dict(
    prompt=
    '(masterpiece:1,2), best quality, masterpiece, best detailed face, a beautiful girl',
    height=512,
    width=512,
    num_inference_steps=1,
    num_images_per_prompt=1,
)

# NOTE: Warm it up.
# The initial calls will trigger compilation and might be very slow.
# After that, it should be very fast.
output_image = model(**kwarg_inputs).images[0]

# # Let's see it!
# # Note: Progress bar might work incorrectly due to the async nature of CUDA.
# begin = time.time()
# output_image = model(**kwarg_inputs).images[0]
# print(f'Inference time: {time.time() - begin:.3f}s')

# # Let's view it in terminal!
# from sfast.utils.term_image import print_image

# print_image(output_image, max_width=80)