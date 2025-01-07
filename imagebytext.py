import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login

# Replace 'your_huggingface_token' with your actual Hugging Face token
huggingface_token = "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"
login(token=huggingface_token)  # Pass the token directly

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("capybara.png")