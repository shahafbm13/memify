from PIL import Image, ImageDraw, ImageFont
import os
import math

def split_text_into_two_lines(text):
    """
    Splits the text into two lines between words, aiming for a balanced split.

    Parameters:
    - text: The input text to split.

    Returns:
    - A tuple: (first_text, second_text)
    """
    words = text.split()
    mid_index = math.ceil(len(words) / 2)

    # Adjust mid_index to ensure the split occurs at a word boundary
    first_text = " ".join(words[:mid_index]).strip()
    second_text = " ".join(words[mid_index:]).strip()

    return first_text, second_text

def split_meme_text(text):
    # Try splitting by a delimiter like a comma first
    if ',' in text:
        upper_text, lower_text = text.split(',', 1)
    # If no comma, split by the first space roughly in the middle
    else:
        words = text.split()
        mid_index = len(words) // 2
        upper_text = ' '.join(words[:mid_index])
        lower_text = ' '.join(words[mid_index:])

    # Trim leading/trailing spaces
    return upper_text.strip(), lower_text.strip()


def calculate_text_split_and_position(
    text, 
    image_size, 
    font_path, 
    max_font_size=100, 
    margin=20
):
    """
    Splits the text into two lines, calculates positions, and an appropriate font size
    for the given image size.

    Parameters:
    - text: The text to be split and positioned.
    - image_size: Tuple (width, height) of the image.
    - font_path: Path to the font file.
    - max_font_size: Maximum allowable font size.
    - margin: Minimum distance from the edges of the image.

    Returns:
    - A tuple: (font_size, first_text_position, second_text_position, first_text, second_text)
    """
    width, height = image_size

    # Split the text into two lines
    first_text, second_text = split_text_into_two_lines(text)

    # Determine the maximum font size that fits within the image width
    font_size = max_font_size
    while font_size > 10:  # Minimum font size threshold
        font = ImageFont.truetype(font_path, font_size)
        first_text_bbox = font.getbbox(first_text)
        second_text_bbox = font.getbbox(second_text)

        first_text_width = first_text_bbox[2] - first_text_bbox[0]
        second_text_width = second_text_bbox[2] - second_text_bbox[0]

        if first_text_width <= width - 2 * margin and second_text_width <= width - 2 * margin:
            break
        font_size -= 1

    if font_size == 10:
        raise ValueError("Text too large to fit in the image even with the smallest font size.")

    # Calculate positions
    font = ImageFont.truetype(font_path, font_size)
    first_text_bbox = font.getbbox(first_text)
    second_text_bbox = font.getbbox(second_text)

    first_text_width = first_text_bbox[2] - first_text_bbox[0]
    first_text_height = first_text_bbox[3] - first_text_bbox[1]

    second_text_width = second_text_bbox[2] - second_text_bbox[0]
    second_text_height = second_text_bbox[3] - second_text_bbox[1]

    first_text_position = ((width - first_text_width) // 2, margin)
    second_text_position = ((width - second_text_width) // 2, height - second_text_height - margin)

    return font_size, first_text_position, second_text_position, first_text, second_text


def add_clear_text_with_outline_to_gif(
    input_gif,
    output_gif,
    text,
    font_path,
    font_size,
    first_text_position,
    second_text_position,
    outline_color=(255, 255, 255),  # White outline
    text_color=(0, 0, 0),  # Black text
    outline_thickness=2  # Thickness of the outline
):
    # Ensure the font file exists
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font file '{font_path}' not found.")
    
    # Load the GIF
    try:
        img = Image.open(input_gif)
    except Exception as e:
        raise IOError(f"Failed to open input GIF: {e}")
    
    frames = []

    def draw_text_with_outline(draw, position, text, font, outline_color, text_color, thickness):
        x, y = position
        # Draw outline
        for dx in range(-thickness, thickness + 1):
            for dy in range(-thickness, thickness + 1):
                if dx != 0 or dy != 0:  # Avoid drawing over the text itself
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        # Draw main text
        draw.text(position, text, font=font, fill=text_color)

    font = ImageFont.truetype(font_path, font_size)

    first_text, second_text = split_meme_text(text)
    
    for i in range(img.n_frames):
        img.seek(i)
        frame = img.convert("RGBA")
        frame_copy = frame.copy()
        draw = ImageDraw.Draw(frame_copy)
        
        # Draw text with outline
        draw_text_with_outline(draw, first_text_position, first_text, font, outline_color, text_color, outline_thickness)
        draw_text_with_outline(draw, second_text_position, second_text, font, outline_color, text_color, outline_thickness)
        
        frames.append(frame_copy)

    for i, frame in enumerate(frames):
        frame.save(f"/gpfs0/bgu-benshimo/users/guyperet/memify/frames/frame_{i:03d}_with_text.png")
        
    # Save the modified GIF
    os.system(f"ffmpeg -y -framerate 14 -i /gpfs0/bgu-benshimo/users/guyperet/memify/frames/frame_%03d_with_text.png {output_gif}")

