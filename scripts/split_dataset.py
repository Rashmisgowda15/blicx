import os
import shutil
import random
from pathlib import Path

def split_dataset(input_dir, output_root, split_ratio=0.8):
    input_path = Path(input_dir)
    output_path = Path(output_root)
    
    # YOLO structure
    train_img_dir = output_path / "train" / "images"
    train_lbl_dir = output_path / "train" / "labels"
    val_img_dir = output_path / "val" / "images"
    val_lbl_dir = output_path / "val" / "labels"
    
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    image_files = list(input_path.glob("*.jpg"))
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    def move_files(files, img_dest, lbl_dest):
        for f in files:
            shutil.copy(str(f), str(img_dest / f.name))
            # Create empty label file for now (placeholder for YOLO)
            label_name = f.stem + ".txt"
            with open(lbl_dest / label_name, 'w') as lf:
                pass # Empty
                
    move_files(train_files, train_img_dir, train_lbl_dir)
    move_files(val_files, val_img_dir, val_lbl_dir)
    
    print(f"Dataset split complete:")
    print(f"Train: {len(train_files)} images")
    print(f"Val: {len(val_files)} images")

if __name__ == "__main__":
    src_dir = r"c:\Users\Rashmi Mithun\Desktop\new dental pro\data\augmented"
    dest_dir = r"c:\Users\Rashmi Mithun\Desktop\new dental pro\data\yolo_setup"
    
    split_dataset(src_dir, dest_dir)
