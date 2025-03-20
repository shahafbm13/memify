#!/usr/bin/env python3

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import login
import google.generativeai as genai

# Diffusers pipelines
from diffusers import DiffusionPipeline, StableVideoDiffusionPipeline

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
# (Optional) set environment variable to reduce fragmentation
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 1) Configure your Google Generative AI API key
GENAI_API_KEY = "INSERT_YOUR_API_KEY"  # e.g. "AIzaSyCP..."
genai.configure(api_key=GENAI_API_KEY)

# 2) Configure your Hugging Face token
HUGGINGFACE_TOKEN = "INSERT_YOUR_HF_TOKEN"  # e.g. "hf_abc123..."
login(token=HUGGINGFACE_TOKEN)

# -------------------------------------------------------
# Utility: generate GIF from a single image using
# StableVideoDiffusionPipeline
# -------------------------------------------------------
def generate_gif_from_image(
    image_path: str,
    output_frames_folder: str = "frames",
    output_gif_folder: str = "gifs",
    num_frames: int = 14,
    num_inference_steps: int = 100,
    height: int = 256,
    width: int = 256,
    gif_duration: int = 200,
) -> str:
    """
    Generate a video from an image using StableVideoDiffusionPipeline,
    save frames, and assemble into a GIF.
    """
    print("Loading Stable Video Diffusion pipeline...")
    video_pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        torch_dtype=torch.float16,
        variant="fp16",
    )

    # Move to GPU if available using CPU offload for memory efficiency
    try:
        video_pipe.enable_model_cpu_offload()
    except Exception as e:
        print("Could not enable CPU offload for video pipe:", e)
        if torch.cuda.is_available():
            video_pipe = video_pipe.to("cuda")

    # (Optional) enable xFormers for memory-efficient attention
    try:
        video_pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print("Could not enable xFormers for video pipe:", e)

    # Open the input image
    print("Opening base image...")
    image = Image.open(image_path).convert("RGB")

    # Generate the video frames
    print(f"Generating {num_frames} frames with {num_inference_steps} steps each...")
    with torch.cuda.amp.autocast():
        output_frames = video_pipe(
            image,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
        ).frames  # List of PIL Images

    # Create output directories
    os.makedirs(output_frames_folder, exist_ok=True)
    os.makedirs(output_gif_folder, exist_ok=True)

    # Save frames individually
    print("Saving frames...")
    frame_paths = []
    for i, frame in enumerate(output_frames):
        frame_path = os.path.join(output_frames_folder, f"frame_{i:03d}.png")
        frame.save(frame_path)
        frame_paths.append(frame_path)

    # Assemble into GIF
    gif_path = os.path.join(output_gif_folder, "output.gif")
    print(f"Assembling frames into GIF: {gif_path}...")
    output_frames[0].save(
        gif_path,
        save_all=True,
        append_images=output_frames[1:],
        duration=gif_duration,
        loop=0,  # 0 = loop forever
    )

    # Unload the pipeline and clear CUDA cache
    del video_pipe
    torch.cuda.empty_cache()

    print(f"GIF generated successfully at: {gif_path}")
    return gif_path


# -------------------------------------------------------
# Main script
# -------------------------------------------------------
if __name__ == "__main__":
    # 1) Ask user for text prompt
    user_prompt = input("Enter a text prompt to generate a GIF: ")

    # 2) Use Generative AI to create a single-sentence meme text
    print("Generating meme-style text via Gemini...")
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        resp1 = model.generate_content(
            f"convert me the text to one meme-style text(your output is only one sentence): '{user_prompt}'"
        )
        meme_text = resp1.text.strip()
    except Exception as e:
        print("Error generating meme text:", e)
        meme_text = user_prompt

    print(f"Generated meme text: {meme_text}")

    # 3) Generate a short meme template description
    print("Generating short meme template description...")
    try:
        resp2 = model.generate_content(
            f"Describe in short (without explains) for me just 1 meme template (image) for the text: '{meme_text}'"
        )
        meme_desc = resp2.text.strip()
    except Exception as e:
        print("Error generating meme template description:", e)
        meme_desc = meme_text

    print(f"Meme template description: {meme_desc}")

    # 4) Load Stable Diffusion pipeline (SDXL) to generate an image
    print("Loading SDXL pipeline for image generation...")
    try:
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
    except Exception as e:
        print("Could not load SDXL pipeline:", e)
        exit(1)

    # 5) Generate the meme image
    print("Generating meme image...")
    try:
        with torch.amp.autocast("cuda"):
            image = pipe(meme_desc).images[0]
    except Exception as e:
        print("Error during image generation:", e)
        image = Image.new("RGB", (256, 256), "white")

    # Show the generated image (comment out if running on a non-GUI machine)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    # Save the image
    image_path = "meme_template.png"
    image.save(image_path)
    print(f"Image saved to {image_path}")

    # Unload SD pipeline to save memory
    del pipe
    torch.cuda.empty_cache()

    # 6) Now generate a GIF from the saved image using Stable Video Diffusion
    output_gif_path = generate_gif_from_image(
        image_path,
        output_frames_folder="frames",
        output_gif_folder="gifs",
        num_frames=14,            # Adjust to your preference
        num_inference_steps=100,  # Adjust to your preference
        height=256,
        width=256,
        gif_duration=200,
    )

    print(f"All done! Final GIF at: {output_gif_path}")






