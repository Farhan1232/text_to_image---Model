# Install required libraries


from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from IPython.display import display

# Load a smaller pre-trained Stable Diffusion model for faster generation
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cpu")  # Use CPU

# Reduce the number of inference steps for faster generation
prompt = "A serene landscape with mountains and a river under a starry sky"
num_inference_steps = 25  # Reduce from default 50 for faster generation

# Generate image
print("Generating image... This will be faster with fewer steps.")
image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]

# Display the image inline in the notebook
display(image)

# Save the image locally
image.save("fast_generated_image.png")
print("Image saved as 'fast_generated_image.png'")
