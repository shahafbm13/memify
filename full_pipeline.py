from PIL import Image
import os
import torch
from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt
from huggingface_hub import login

def generate_gif_from_image(
    image_path,
    output_frames_folder="frames",
    output_gif_folder="gifs",
    num_frames=14,
    num_inference_steps=150,
    height=512,
    width=512,
    gif_duration=200
):
    """
    Generate a video from an image using StableVideoDiffusionPipeline, save frames to a folder,
    and assemble them into a GIF saved in another folder.

    Args:
        image_path (str): Path to the input image.
        output_frames_folder (str): Directory to save individual frames.
        output_gif_folder (str): Directory to save the resulting GIF.
        num_frames (int): Number of frames to generate.
        num_inference_steps (int): Number of inference steps for the model.
        height (int): Height of the generated frames.
        width (int): Width of the generated frames.
        gif_duration (int): Duration of each frame in the GIF (in milliseconds).

    Returns:
        str: Path to the generated GIF.
    """
    # Load the pipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", 
        torch_dtype=torch.float16,  # Use float16 to save memory
        variant="fp16"
    )

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # Enable memory efficient attention and VAE slicing
    pipe.enable_model_cpu_offload()

    # Open the image
    image = Image.open(image_path).convert("RGB")

    # Generate the video frames
    with torch.cuda.amp.autocast():  # Enable automatic mixed precision
        output = pipe(
            image, 
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
        ).frames  # This is a list of PIL Images

    # Create directories for frames and GIF if they don't exist
    os.makedirs(output_frames_folder, exist_ok=True)
    os.makedirs(output_gif_folder, exist_ok=True)

    # Save individual frames
    frame_paths = []
    for i, frame in enumerate(output):
        frame_path = os.path.join(output_frames_folder, f"frame_{i:03d}.png")
        frame.save(frame_path)
        frame_paths.append(frame_path)

    # Save the frames as a GIF
    gif_path = os.path.join(output_gif_folder, "output.gif")
    output[0].save(
        gif_path,
        save_all=True,
        append_images=output[1:],
        duration=gif_duration,
        loop=0  # 0 = loop forever
    )

    # Clear CUDA cache
    torch.cuda.empty_cache()

    print(f"GIF generated successfully and saved to {gif_path}")
    return gif_path

import google.generativeai as genai

genai.configure(api_key="AIzaSyCP15lhd_MQ108q3j_xSyoyE0s7RmMUx-I")
model = genai.GenerativeModel("gemini-1.5-flash")


# Generate a meme template description from user input
user_prompt = input("Enter a text prompt to generate a GIF: ")

response = model.generate_content(f"convert me the text to one meme-style text(your output is only one sentence): '{user_prompt}'")
print(f"Generated meme template description: {response.text}")
response = model.generate_content(f"Describe in short (without explains) for me just an 1 meme template (image) for the text: '{response.text}'")
print(f"Generated meme template description: {response.text}")

import torch
from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt
from huggingface_hub import login

# Replace 'your_huggingface_token' with your actual Hugging Face token
huggingface_token = "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"
login(token=huggingface_token)  # Pass the token directly

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

image = pipe(response.text).images[0]

# Or, if you're using a Jupyter notebook or Google Colab, you can also use matplotlib to display it inline:
plt.imshow(image)
plt.axis('off')  # Hide the axis
plt.show()

# Save the image to a file
image.save("meme_template.png")

# Create gif from image
output_gif_path = generate_gif_from_image("meme_template.png")



