import zipfile
import os
from pathlib import Path

def extract_dataset(zip_path, extract_path):
    # Ensure extract path exists
    os.makedirs(extract_path, exist_ok=True)
    
    print(f"Extracting {zip_path} to {extract_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction complete.")
    except Exception as e:
        print(f"Error extracting dataset: {e}")

if __name__ == "__main__":
    ZIP_PATH = r"C:\Users\Rashmi Mithun\Downloads\Periapical Images.zip"
    EXTRACT_PATH = r"c:\Users\Rashmi Mithun\Desktop\new dental pro\data\raw\Periapical Images"
    extract_dataset(ZIP_PATH, EXTRACT_PATH)
