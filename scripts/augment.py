import cv2
import os
import random
import numpy as np
from pathlib import Path

def augment_image(image):
    augmentations = []
    
    # 1. Original
    augmentations.append(image)
    
    # 2. Horizontal Flip
    augmentations.append(cv2.flip(image, 1))
    
    # 3. Rotation (Small angles)
    for angle in [-15, -10, 5, 10, 15]:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmentations.append(rotated)
    
    # 4. Brightness/Contrast
    alpha = 1.2 # Contrast
    beta = 10   # Brightness
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    augmentations.append(adjusted)
    
    # 5. Gaussian Blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    augmentations.append(blurred)
    
    return augmentations

def process_augmentation(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_path.glob("*.jpg"))
    total_generated = 0
    
    for img_file in image_files:
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        aug_versions = augment_image(img)
        
        base_name = img_file.stem
        for i, aug_img in enumerate(aug_versions):
            save_name = f"{base_name}_aug_{i}.jpg"
            cv2.imwrite(str(output_path / save_name), aug_img)
            total_generated += 1
            
    print(f"Augmentation complete. Total images generated: {total_generated}")

if __name__ == "__main__":
    PROCESSED_DIR = r"c:\Users\Rashmi Mithun\Desktop\new dental pro\data\processed"
    AUGMENTED_DIR = r"c:\Users\Rashmi Mithun\Desktop\new dental pro\data\augmented"
    
    process_augmentation(PROCESSED_DIR, AUGMENTED_DIR)
