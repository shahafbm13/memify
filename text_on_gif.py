from PIL import Image, ImageDraw, ImageFont
import os
from textwrap import wrap

def split_text_for_top_and_bottom(text):
    """
    Splits the text into two parts for the top and bottom of the image.
    Tries to split at a logical breakpoint near the middle.

    Parameters:
    - text: The full text to split.

    Returns:
    - A tuple: (top_text, bottom_text)
    """
    words = text.split()
    mid_index = len(words) // 2

    # Split at the middle word
    top_text = " ".join(words[:mid_index]).strip()
    bottom_text = " ".join(words[mid_index:]).strip()

    return top_text, bottom_text

def split_text_into_lines(text, font, max_width):
    """
    Splits the text into multiple lines to fit within the specified width.

    Parameters:
    - text: The text to split.
    - font: The font object used to measure text size.
    - max_width: The maximum allowable width for a line.

    Returns:
    - A list of text lines.
    """
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        # Check if adding the word exceeds the width
        test_line = " ".join(current_line + [word])
        if font.getbbox(test_line)[2] <= max_width:
            current_line.append(word)
        else:
            # Current line is complete, add to lines and start a new line
            lines.append(" ".join(current_line))
            current_line = [word]

    # Add the last line if it exists
    if current_line:
        lines.append(" ".join(current_line))

    return lines

def calculate_text_positions_and_font_size(
    text, image_size, font_path, max_font_size=100, margin=20, position="top"
):
    """
    Adjust font size and calculate positions for multiple lines of text.

    Parameters:
    - text: The input text to fit within the image.
    - image_size: Tuple (width, height) of the image.
    - font_path: Path to the font file.
    - max_font_size: Maximum allowable font size.
    - margin: Minimum distance from the edges of the image.
    - position: Either "top" or "bottom" for text placement.

    Returns:
    - font_size, positions (list of (x, y)), lines (list of text lines).
    """
    width, height = image_size

    # Find the largest font size that works
    font_size = max_font_size
    while font_size > 10:
        font = ImageFont.truetype(font_path, font_size)
        max_line_width = width - 2 * margin
        lines = split_text_into_lines(text, font, max_line_width)

        # Total height of text block
        total_text_height = sum(
            [font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines]
        ) + (len(lines) - 1) * margin

        if total_text_height <= height / 2 - margin:  # Ensure it fits within half the image
            break
        font_size -= 1

    if font_size == 10:
        raise ValueError("Text is too large to fit the image.")

    # Calculate positions for each line
    positions = []
    if position == "top":
        y_offset = margin  # Start at the top margin
    elif position == "bottom":
        y_offset = height - total_text_height - 2 * margin  # Start above the bottom margin
    else:
        raise ValueError("Invalid position. Use 'top' or 'bottom'.")

    for line in lines:
        line_width = font.getbbox(line)[2] - font.getbbox(line)[0]
        x_position = (width - line_width) // 2  # Center text horizontally
        positions.append((x_position, y_offset))
        y_offset += font.getbbox(line)[3] - font.getbbox(line)[1] + margin

    return font_size, positions, lines

def draw_text_with_outline(draw, position, text, font, outline_color, text_color, thickness):
    x, y = position
    # Draw outline
    for dx in range(-thickness, thickness + 1):
        for dy in range(-thickness, thickness + 1):
            if dx != 0 or dy != 0:  # Avoid overwriting the main text
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    # Draw main text
    draw.text(position, text, font=font, fill=text_color)

def add_clear_text_with_outline_to_gif(
    input_gif,
    output_gif,
    text,
    font_path,
    outline_color=(0, 0, 0),
    text_color=(255, 255, 255),
    outline_thickness=1,
    max_font_size=30,
    margin=2,
):
    # Split the text into top and bottom parts dynamically
    top_text, bottom_text = split_text_for_top_and_bottom(text)

    # Load GIF
    img = Image.open(input_gif)
    frames = []

    # Calculate font size and positions for top and bottom text
    top_font_size, top_positions, top_lines = calculate_text_positions_and_font_size(
        top_text, img.size, font_path, max_font_size, margin, position="top"
    )
    bottom_font_size, bottom_positions, bottom_lines = calculate_text_positions_and_font_size(
        bottom_text, img.size, font_path, max_font_size, margin, position="bottom"
    )

    top_font = ImageFont.truetype(font_path, top_font_size)
    bottom_font = ImageFont.truetype(font_path, bottom_font_size)

    for i in range(img.n_frames):
        img.seek(i)
        frame = img.convert("RGBA")
        frame_copy = frame.copy()
        draw = ImageDraw.Draw(frame_copy)

        # Draw top text with outline
        for line, position in zip(top_lines, top_positions):
            draw_text_with_outline(draw, position, line, top_font, outline_color, text_color, outline_thickness)

        # Draw bottom text with outline
        for line, position in zip(bottom_lines, bottom_positions):
            draw_text_with_outline(draw, position, line, bottom_font, outline_color, text_color, outline_thickness)

        frames.append(frame_copy)

    # Save frames as GIF
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=img.info.get("duration", 100),
    )
    
    
# if __name__ == "__main__":
#     # Example usage
#     gif_path = "/gpfs0/bgu-benshimo/users/guyperet/memify/gif_outputs_try5/output_gif_2.gif"
#     meme_text = "What I am trying to do when the freezer get stuck, is basically the opposite of the freezer."
#     font_path = "/gpfs0/bgu-benshimo/users/guyperet/memify/Avita-Black.otf"
#     add_clear_text_with_outline_to_gif(gif_path, gif_path, meme_text, font_path)
