"""
Prepare dataset for end-to-end ViT training.
Instead of pre-extracting features, we save image paths and labels.
The model will load and process panel crops on-the-fly.
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_ROOT = "/root/autodl-tmp/LlamaFactory/data/ccac2025_complete/task1"
IMAGE_ROOT = "/root/autodl-tmp/LlamaFactory/data/ccac2025_complete"
SAVE_ROOT = "./data_e2e"


def split_image(image):
    """Split 2x2 image into 4 panels (top-left, top-right, bottom-left, bottom-right)."""
    w, h = image.size
    panels = [
        image.crop((0, 0, w // 2, h // 2)),
        image.crop((w // 2, 0, w, h // 2)),
        image.crop((0, h // 2, w // 2, h)),
        image.crop((w // 2, h // 2, w, h)),
    ]
    return panels


def parse_label(label_str):
    """Parse '0 1 2 3' into [0,1,2,3]."""
    return [int(x) for x in label_str.strip().split()]


def process_split(split_name, json_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    records = []
    for item in tqdm(data, desc=f"Processing {split_name}"):
        img_rel = item["images"][0]
        img_path = os.path.join(IMAGE_ROOT, img_rel)
        
        prompt = item["conversations"][0]["value"]
        label_str = item["conversations"][1]["value"]
        label_seq = parse_label(label_str)
        
        # Determine language from path or prompt
        lang = "zh" if "zh" in img_rel else "en"
        
        sample_name = os.path.splitext(os.path.basename(img_rel))[0]
        
        # Save 4 panel crops
        image = Image.open(img_path).convert("RGB")
        panels = split_image(image)
        panel_paths = []
        for idx, panel in enumerate(panels):
            panel_path = os.path.join(save_dir, f"{sample_name}_panel{idx}.jpg")
            panel.save(panel_path)
            panel_paths.append(panel_path)
        
        records.append({
            "sample_name": sample_name,
            "panel_paths": panel_paths,
            "label_seq": label_seq,
            "label_str": label_str,
            "lang": lang,
            "prompt": prompt,
        })
    
    # Save records
    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(records)} samples to {save_dir}")


def main():
    splits = [
        ("train", os.path.join(DATA_ROOT, "train/train.json"), os.path.join(SAVE_ROOT, "train")),
        ("valid", os.path.join(DATA_ROOT, "valid/valid.json"), os.path.join(SAVE_ROOT, "valid")),
        ("test", os.path.join(DATA_ROOT, "test/test.json"), os.path.join(SAVE_ROOT, "test")),
    ]
    
    for split_name, json_path, save_dir in splits:
        process_split(split_name, json_path, save_dir)


if __name__ == "__main__":
    main()
