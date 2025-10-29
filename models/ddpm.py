# !pip install diffusers
from diffusers import DDPMPipeline

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)

# run pipeline in inference (sample random noise and denoise)
image = ddpm().images[0]

# save image
image.save("ddpm_generated_image.png")
