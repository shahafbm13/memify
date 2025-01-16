from diffusers import StableVideoDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import torch
from sentence_transformers import SentenceTransformer
import os
import re
import numpy as np
from huggingface_hub import login
from transformers import pipeline
from text_on_gif import *
from text2vec import *
from argparse import ArgumentParser

device = "cuda" if torch.cuda.is_available() else "cpu"
hf_token = "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"
login(token=hf_token)
pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct")
sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
img2vid = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16)
img2vid = img2vid.to(device)
img2vid.enable_model_cpu_offload()

def meme_text_generator(user_prompt: str):
    """
    Generate a meme-like text based on the user prompt.
    """
    messages = [
    {"role": "user", "content": 
        f"""**Instruction:**
                You are a creative and witty AI. Please write a short, meme-like one-liner about the subject "{user_prompt}".
                Keep it concise—one or two short sentences only—and make it humorous or engaging in the style of an internet meme.

            **Goal:**
                Produce a fun, meme-like phrase referencing "{user_prompt}".
                Keep it short and funny.
                You may include pop-culture references or comedic exaggeration.

            **Format:**
                One or two short sentences maximum.
                Meme-like.

            Begin now.
            """
    }]
    response = pipe(messages, return_full_text=False, truncation=False, max_new_tokens=50)
    return response[0]["generated_text"]

def meme_template_picker(meme_text: str):
    """
    Pick a meme template based on the meme text.
    """
    with open("/gpfs0/bgu-benshimo/users/guyperet/memify/meme_text_description_file.json", "r", encoding="utf-8") as f:
        meme_data = json.load(f)
        
    
    result = find_best_meme_description(meme_text, meme_data, sentence_encoder)
    print(f'Best meme matched with the text is {result["best_meme_path"]} with similarity score of {result["similarity"]}')
    return result["best_meme_path"]
    

def gif_generator(meme_template_path: str, index : int):
    """
    Generate a GIF based on the meme template.
    """
    image = Image.open(meme_template_path).convert("RGB") # ? Normalize the image
    # Generate the video with reduced memory usage
    with torch.amp.autocast('cuda'):  # Enable automatic mixed precision
        output = img2vid(
            image, 
            num_frames=14,  # Reduce number of frames if needed
            num_inference_steps=150,  # Reduce number of inference steps
            height=512,  # Reduce height if needed
            width=512,  # Reduce width if needed            
        ).frames
    for i, frame in enumerate(output[0]):
        frame.save(f"/gpfs0/bgu-benshimo/users/guyperet/memify/frames/frame_{i:03d}.png")
        
        

    # Check what is the last index of the gifs in gif_outputs folder, if empty - start from 0 in the format of "output_gif_000.gif"
    gif_path = "/gpfs0/bgu-benshimo/users/guyperet/memify/gif_outputs"
    gif_name = f"output_gif_{index}.gif"
    # os.system(f"ffmpeg -y -framerate 14 -i /gpfs0/bgu-benshimo/users/guyperet/memify/frames/frame_%03d.png {gif_path}/{gif_name}")
    output[0][0].save(f"{gif_path}/{gif_name}", save_all=True, append_images=output[0][1:], duration=100, loop=0)
    # Return full gif_path
    return f'{gif_path}/{gif_name}'
    
    

def text_on_gif(gif_path: str, meme_text: str):
    font_path = "/gpfs0/bgu-benshimo/users/guyperet/memify/Avita-Black.otf"
    # Pass the calculated parameters to the GIF function
    add_clear_text_with_outline_to_gif(
        input_gif=gif_path,
        output_gif=gif_path,
        text=meme_text,
        font_path=font_path,
    )

    

if __name__ == "__main__":
    # # Seed 
    # torch.manual_seed(42)  # For reproducibility
    
    # arg_parser = ArgumentParser()
    # arg_parser.add_argument("-p", "--prompt", type=str, required=True, help="User prompt for meme generation")
    # args = arg_parser.parse_args()
    # user_propmpt = args.prompts
    
    user_prompts = [
    "Mondays",
    "Wi-Fi passwords",
    "coffee addiction",
    "Zoom calls",
    "online shopping",
    "gym memberships",
    "social media influencers",
    "cat videos",
    "morning alarms",
    "Sunday scaries",
    "texting typos",
    "airplane food",
    "road trips",
    "office gossip",
    "self-checkout machines",
    "weather apps",
    "password resets",
    "traffic jams",
    "holiday sales",
    "birthday parties",
    "group chats",
    "pet selfies",
    "overpacked luggage",
    "fast food drive-thrus",
    "reality TV",
    "work emails",
    "long queues",
    "last-minute plans",
    "diet fads",
    "oversized hoodies",
    "binge-watching",
    "smartphone battery life",
    "public Wi-Fi",
    "celebrity gossip",
    "cancel culture",
    "pop quizzes",
    "weekend plans",
    "sleeping in",
    "food delivery apps",
    "cryptocurrency",
    "daily horoscopes",
    "rainy days",
    "awkward silences",
    "life hacks",
    "broken phone screens",
    "late-night snacks",
    "houseplants",
    "morning commutes",
    "air conditioner remotes",
    "autocorrect fails",
    "online dating",
    "fitness trackers",
    "wifi signal strength",
    "socks with sandals",
    "New Year resolutions",
    "lazy Sundays",
    "spilled coffee",
    "unread emails",
    "unexpected guests",
    "forgotten passwords",
    "selfie sticks",
    "early morning flights",
    "caffeine crashes",
    "holiday gift wrapping",
    "unwanted notifications",
    "dancing in the rain",
    "weekend hangovers",
    "charging cables",
    "daily commutes",
    "overpriced coffee",
    "phone storage limits",
    "fast fashion",
    "budget airlines",
    "rainbow after the storm",
    "pizza delivery delays",
    "sunny beach days",
    "group photo fails",
    "misheard song lyrics",
    "karaoke nights",
    "ice cream cravings",
    "rollercoaster rides",
    "game night arguments",
    "vacation photos",
    "procrastination",
    "travel bucket lists",
    "birthday surprises",
    "midnight cravings",
    "unexpected rain",
    "street food adventures",
    "dog walks",
    "board game nights",
    "silent phone calls",
    "clumsy moments",
    "movie marathons",
    "morning jogs",
    "skipping ads",
    "shopping sprees",
    "midweek blues"
    ]
    for i, user_propmpt in enumerate(user_prompts):
        # Pass the user propmpt to the meme-text generator
        meme_text = meme_text_generator(user_propmpt)
        print(f'Meme text: {meme_text}')
        # Pick a meme template based on usern propmt
        meme_template_path = meme_template_picker(meme_text)
        print(f'Meme template: {meme_template_path}')
        # Animate the meme template
        gif_path = gif_generator(meme_template_path, i)
        print(f'GIF path: {gif_path}')
        # # Add meme text to the GIF
        text_on_gif(gif_path, meme_text)
        print(f'Done with gif #{i}')
        # Empty the frames folder
        frames_path = "/gpfs0/bgu-benshimo/users/guyperet/memify/frames"
        files = os.listdir(frames_path)
        for file in files:
            os.remove(f"{frames_path}/{file}")
        
    print("Done with all the GIFs")