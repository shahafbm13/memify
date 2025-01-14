from diffusers import StableVideoDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import torch
from gensim.models import KeyedVectors
import os
import re
import numpy as np
from huggingface_hub import login
from transformers import pipeline
from text_on_gif import *
from text2vec import *
from argparse import ArgumentParser


hf_token = "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"
login(token=hf_token)
pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct")

def meme_text_generator(user_prompt: str):
    """
    Generate a meme-like text based on the user prompt.
    """
    messages = [
    {"role": "user", "content": f"Provide a meme-like text for the following description: '{user_prompt}'"},
    ]
    response = pipe(messages)
    return response[0]["generated_text"][1]["content"]

def meme_template_picker(meme_text: str):
    """
    Pick a meme template based on the meme text.
    """
    with open("/home/ubuntu/GenAI/meme_text_description.json", "r", encoding="utf-8") as f:
        meme_data = json.load(f)
        
    meme_template_path = "/home/ubuntu/GenAI/meme_templates/meme_templates"
    meme_names = os.listdir(meme_template_path)
    # Clean meme names in the template folder
    cleaned_meme_names = [clean_meme_name(name) for name in meme_names]
    # Filter meme_data to only include memes present in the template folder
    meme_data = {
        meme_name: descriptions
        for meme_name, descriptions in meme_data.items()
        if clean_meme_name(meme_name) in cleaned_meme_names
    }
    
    model_path = "GoogleNews-vectors-negative300.bin"
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    
    result = find_best_meme_description(meme_text, meme_data, model)
    result = result["best_meme_name"].replace(" ", "_") + ".jpg"
    for i, meme_path in enumerate(meme_names):
        if result in meme_path:
            return meme_path
    

    


def download_template(meme_template_name: str):
    """
    Download the meme template based on the name.
    """
    pass

def gif_generator(meme_template_name: str):
    """
    Generate a GIF based on the meme template.
    """
    pass

def text_on_gif(gif_path: str, meme_text: str):
    output_gif = "/home/ubuntu/GenAI/gif_outputs/with_text.gif"
    font_path = "/home/ubuntu/GenAI/fonts/Avita-Black.otf"

    # Load the GIF to get its dimensions
    with Image.open(gif_path) as gif:
        image_size = gif.size

    # Calculate font size and positions
    font_size, first_text_position, second_text_position, first_text, second_text = calculate_text_split_and_position(
        meme_text, image_size, font_path, max_font_size=30, margin=20
    )

    # Pass the calculated parameters to the GIF function
    add_clear_text_with_outline_to_gif(
        input_gif=input_gif,
        output_gif=output_gif,
        text=f"{first_text},{second_text}",
        font_path=font_path,
        font_size=font_size,
        first_text_position=first_text_position,
        second_text_position=second_text_position,
        outline_color=(0, 0, 0),
        text_color=(255, 255, 255),
        outline_thickness=2
    )

    

if __name__ == "__main__":
    # Seed 
    torch.manual_seed(42)  # For reproducibility
    
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-p", "--prompt", type=str, required=True, help="User prompt for meme generation")
    args = arg_parser.parse_args()
    user_propmpt = args.prompt
    # Pass the user propmpt to the meme-text generator
    meme_text = meme_text_generator(user_propmpt)
    # Pick a meme template based on usern propmt
    meme_template_name = meme_template_picker(meme_text)
    # Download/Get the image based on the name of the meme template
    print(f'Path: {meme_template_name}')
    assert 0
    image = download_template(meme_template_name)
    
    # Animate the meme template
    gif_path = gif_generator(meme_template_name)
    
    # Add meme text to the GIF
    text_on_gif(gif_path, meme_text)
    
    # Read the final output