#!/usr/bin/env python3
"""
Convert a PNG image to a smaller JPEG for viewing
"""

from PIL import Image
import sys

def convert_image(input_path, output_path, quality=30):
    """Convert a PNG to a JPEG with specified quality"""
    img = Image.open(input_path)
    
    # Resize if needed
    MAX_SIZE = (800, 800)
    img.thumbnail(MAX_SIZE, Image.LANCZOS)
    
    # Save as JPEG
    img.convert("RGB").save(output_path, "JPEG", quality=quality)
    print(f"Converted {input_path} to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_image.py input_path output_path")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    convert_image(input_path, output_path)