"""
Prepare dataset for multimodal panel order prediction.
Split 2x2 comic into 4 panels and extract text prompts.
"""
import json
import os
from PIL import Image
import numpy as np
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from datasets import Dataset, DatasetDict
from tqdm import tqdm


def parse_and_split(json_path, image_root, split_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for idx, item in enumerate(data):
        img_path = item["images"][0]
        full_path = os.path.join(image_root, img_path)
        prompt = item["conversations"][0]["value"].replace("<image>", "").strip()
        answer = item["conversations"][1]["value"].strip()
        if answer == "" and "test" not in json_path:
            continue
        
        # Determine language from path
        lang = "unknown"
        if "/zh/" in img_path or "/zh_split/" in img_path:
            lang = "zh"
        elif "/en/" in img_path or "/en_split/" in img_path:
            lang = "en"
        
        img = Image.open(full_path).convert("RGB")
        w, h = img.size
        panels = {
            0: img.crop((0, 0, w//2, h//2)),
            1: img.crop((w//2, 0, w, h//2)),
            2: img.crop((0, h//2, w//2, h)),
            3: img.crop((w//2, h//2, w, h)),
        }
        sample_dir = os.path.join(split_dir, f"sample_{idx:05d}")
        os.makedirs(sample_dir, exist_ok=True)
        panel_paths = {}
        for p_idx, p_img in panels.items():
            p_path = os.path.join(sample_dir, f"panel_{p_idx}.jpg")
            p_img.save(p_path)
            panel_paths[p_idx] = p_path
        
        label_seq = None
        if answer != "":
            label_seq = [int(x) for x in answer.split()]
        
        records.append({
            "sample_dir": sample_dir,
            "sample_name": os.path.basename(sample_dir),
            "panel_0": panel_paths[0],
            "panel_1": panel_paths[1],
            "panel_2": panel_paths[2],
            "panel_3": panel_paths[3],
            "prompt": prompt,
            "lang": lang,
            "label_str": answer,
            "label_seq": label_seq,
        })
    return records


def extract_vit_features(records, out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(device)
    model.eval()
    transform = ViT_B_16_Weights.DEFAULT.transforms()
    
    batch_size = 32
    all_items = []
    for r in records:
        for p_idx in range(4):
            all_items.append((r["sample_dir"], p_idx, r[f"panel_{p_idx}"]))
    
    with torch.no_grad():
        for i in tqdm(range(0, len(all_items), batch_size), desc="Extracting ViT features"):
            batch = all_items[i:i+batch_size]
            imgs = [transform(Image.open(p).convert("RGB")) for _, _, p in batch]
            x = torch.stack(imgs).to(device)
            x = model._process_input(x)
            n = x.shape[0]
            cls_token = model.class_token.expand(n, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = model.encoder(x)
            feats = x[:, 0].cpu().numpy()
            
            for j, (sample_dir, p_idx, _) in enumerate(batch):
                sample_name = os.path.basename(sample_dir)
                feat_path = os.path.join(out_dir, f"{sample_name}.npy")
                if os.path.exists(feat_path):
                    arr = np.load(feat_path)
                else:
                    arr = np.zeros((4, 768), dtype=np.float32)
                arr[p_idx] = feats[j]
                np.save(feat_path, arr)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(script_dir, "../../LlamaFactory/data/ccac2025_complete/task1")
    image_root = os.path.join(script_dir, "../../LlamaFactory/data/ccac2025_complete")
    split_dir = os.path.join(script_dir, "../data_split/panels")
    os.makedirs(split_dir, exist_ok=True)
    
    train_records = parse_and_split(os.path.join(root, "train/train.json"), image_root, os.path.join(split_dir, "train"))
    valid_records = parse_and_split(os.path.join(root, "valid/valid.json"), image_root, os.path.join(split_dir, "valid"))
    test_records = parse_and_split(os.path.join(root, "test/test.json"), image_root, os.path.join(split_dir, "test"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Extracting ViT features on", device)
    extract_vit_features(train_records, os.path.join(script_dir, "../data_split/features/train"), device)
    extract_vit_features(valid_records, os.path.join(script_dir, "../data_split/features/valid"), device)
    extract_vit_features(test_records, os.path.join(script_dir, "../data_split/features/test"), device)
    
    all_prompts = [r["prompt"] for r in train_records + valid_records + test_records]
    vocab = {"<pad>": 0, "<unk>": 1}
    for prompt in all_prompts:
        for word in prompt.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    vocab_path = os.path.join(script_dir, "../data/vocab.json")
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocab size: {len(vocab)}")
    
    data_dir = os.path.join(script_dir, "../data")
    os.makedirs(data_dir, exist_ok=True)
    
    def to_hf(records, has_label=True):
        return Dataset.from_dict({
            "sample_name": [r["sample_name"] for r in records],
            "lang": [r["lang"] for r in records],
            "prompt": [r["prompt"] for r in records],
            "label_str": [r["label_str"] for r in records],
            "label_seq": [r["label_seq"] if has_label else [-1,-1,-1,-1] for r in records],
        })
    
    dataset = DatasetDict({
        "train": to_hf(train_records, True),
        "validation": to_hf(valid_records, True),
        "test": to_hf(test_records, False),
    })
    dataset.save_to_disk(os.path.join(data_dir, "dataset"))
    print(f"Saved dataset. Train:{len(train_records)} Valid:{len(valid_records)} Test:{len(test_records)}")


if __name__ == "__main__":
    main()
