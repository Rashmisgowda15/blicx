import os
import shutil
import random
from pathlib import Path

def organize_dataset(augmented_dir, processed_dir, dest_root):
    """
    Organizes the augmented images based on labels found in the processed directory.
    """
    aug_path = Path(augmented_dir)
    proc_path = Path(processed_dir)
    dest_path = Path(dest_root)
    
    classes = ['healthy', 'caries']
    
    # Check if user has sorted images in processed_dir
    labeled_data = {}
    for c in classes:
        class_proc_dir = proc_path / c
        if class_proc_dir.exists():
            for f in class_proc_dir.glob("*.jpg"):
                labeled_data[f.stem] = c
    
    # If no labels found, warn user
    if not labeled_data:
        print("WARNING: No labeled folders found in 'data/processed'. Falling back to random labels.")
    
    # Recreate destination structure
    for mode in ['train', 'val']:
        for c in classes:
            dir_path = dest_path / mode / c
            if dir_path.exists(): shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)

    # Get all augmented files and group them by base image
    all_aug_files = list(aug_path.glob("*.jpg"))
    image_groups = {}
    for f in all_aug_files:
        base_id = "_".join(f.stem.split("_")[:2]) # e.g., periapical_0001
        if base_id not in image_groups: image_groups[base_id] = []
        image_groups[base_id].append(f)
    
    base_ids = list(image_groups.keys())
    random.shuffle(base_ids)
    
    # Split by original image to prevent data leakage
    split_idx = int(len(base_ids) * 0.8)
    train_ids = base_ids[:split_idx]
    val_ids = base_ids[split_idx:]
    
    def distribute(ids, mode):
        for base_id in ids:
            label = labeled_data.get(base_id, random.choice(classes))
            for f in image_groups[base_id]:
                shutil.copy(str(f), str(dest_path / mode / label / f.name))
            
    distribute(train_ids, 'train')
    distribute(val_ids, 'val')
    print(f"Organization complete. Base images: {len(base_ids)}, Labels found: {len(labeled_data)}")

if __name__ == "__main__":
    AUG = r"c:\Users\Rashmi Mithun\Desktop\new dental pro\data\augmented"
    PROC = r"c:\Users\Rashmi Mithun\Desktop\new dental pro\data\processed"
    DEST = r"c:\Users\Rashmi Mithun\Desktop\new dental pro\data\classification_setup"
    organize_dataset(AUG, PROC, DEST)
