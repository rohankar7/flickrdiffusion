import os
from config import *
from PIL import Image
from tqdm import tqdm

def resize_images(target_size=(256, 256)):
    os.makedirs(resized_img_dir, exist_ok=True)
    for filename in tqdm(os.listdir(image_dir)[:]):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            try:
                img = Image.open(os.path.join(image_dir, filename))
                # Image.Resampling.LANCZOS or Image.Resampling.BICUBIC
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                img_resized.save(os.path.join(resized_img_dir, filename))
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    resize_images()