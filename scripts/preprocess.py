import cv2
import os
import numpy as np
from pathlib import Path

def preprocess_images(input_dir, output_dir, target_size=(640, 640)):
    """
    Cleans and preprocesses dental X-ray images.
    - Resizes to target_size
    - Converts to grayscale
    - Applies CLAHE for contrast enhancement
    - Normalizes intensity
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions]
    
    print(f"Found {len(image_files)} images. Starting preprocessing...")

    for i, img_file in enumerate(image_files):
        try:
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Skipping corrupted file: {img_file.name}")
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE to enhance dental features (roots, caries, bone)
            enhanced = clahe.apply(gray)

            # Resize
            resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_AREA)

            # Normalize (0-255)
            normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)

            # Save processed image
            new_filename = f"periapical_{i+1:04d}.jpg"
            cv2.imwrite(str(output_path / new_filename), normalized)
            
            if (i+1) % 5 == 0:
                print(f"Processed {i+1}/{len(image_files)} images...")

        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")

    print(f"Preprocessing complete. Processed images saved to: {output_dir}")

if __name__ == "__main__":
    RAW_DIR = r"c:\Users\Rashmi Mithun\Desktop\new dental pro\data\raw\Periapical Images"
    PROCESSED_DIR = r"c:\Users\Rashmi Mithun\Desktop\new dental pro\data\processed"
    
    preprocess_images(RAW_DIR, PROCESSED_DIR)
