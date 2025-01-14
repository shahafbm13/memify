from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import torch
import os

# Seed
torch.manual_seed(42)

# Load the pipeline
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", 
    torch_dtype=torch.float16
)    

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Enable memory efficient attention and VAE slicing
pipe.enable_model_cpu_offload()

# Open the image
image = Image.open("1.jpg").convert("RGB")
# Generate the video with reduced memory usage
with torch.amp.autocast('cuda'):  # Enable automatic mixed precision
    output = pipe(
        image, 
        num_frames=14,  # Reduce number of frames if needed
        num_inference_steps=75,  # Reduce number of inference steps
        height=720,  # Reduce height if needed
        width=720,  # Reduce width if needed
    ).frames
for i, frame in enumerate(output[0]):
    frame.save(f"frames14/frame_{i:03d}.png")
os.system(f"ffmpeg -framerate 14 -i frames14/frame_%03d.png output_test.gif")


# # Save the output frames as a gif
# output[0][0].save("output.gif", save_all=True, append_images=output[1:], duration=1000, loop=0)

# Create a directory to store individual frames

# Save individual frames

# print("Individual frames saved")

# # Use FFmpeg to combine frames into a GIF
# # Make sure FFmpeg is installed on your system
# os.system("ffmpeg -framerate 24 -i frames/frame_%03d.png output.gif")

# print("Video saved as output.gif")

# # Optionally, remove individual frame files to save space
# for i in range(len(output)):
#     os.remove(f"frames/frame_{i:03d}.png")

# # Remove the frames directory
# os.rmdir("frames")

# print("Cleanup completed")

# Clear CUDA cache
torch.cuda.empty_cache()

# print("Process completed")