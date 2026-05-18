"""
v4: End-to-end ViT-B/16 with 22-way permutation classification.

Key changes:
1. Load 4 panel crops on-the-fly.
2. Pass each panel through ViT-B/16, extract [CLS] token.
3. Freeze first 8 ViT blocks, fine-tune the last 4 blocks + head.
4. 22-way classification with cross-entropy loss.
5. Strong regularization (dropout 0.5, weight decay 1e-4).
6. Data augmentation for panels (ColorJitter, RandomHorizontalFlip).
"""

import os
import sys
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_ROOT = "./data_e2e"
OUTPUT_DIR = "./outputs/mm-panel-order-v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load exact 22-class mapping from the original ViT pipeline
with open("/root/autodl-tmp/panel_order_vit_pipeline/data/label_mapping.json", "r") as f:
    _LABEL_MAPPING = json.load(f)
    
ALL_PERMUTATIONS = []
for i in range(22):
    perm_str = _LABEL_MAPPING["id2label"][str(i)]
    ALL_PERMUTATIONS.append([int(x) for x in perm_str.split()])

PERM_TO_IDX = {tuple(p): i for i, p in enumerate(ALL_PERMUTATIONS)}
IDX_TO_PERM = {i: p for i, p in enumerate(ALL_PERMUTATIONS)}


def parse_label(label_str):
    return [int(x) for x in label_str.strip().split()]


def load_metadata(split):
    path = os.path.join(DATA_ROOT, split, "metadata.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ComicPanelDataset(Dataset):
    def __init__(self, records, transform=None, augment=False):
        self.records = records
        self.transform = transform
        self.augment = augment
        
        if augment:
            self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        panels = []
        for ppath in rec["panel_paths"]:
            img = Image.open(ppath).convert("RGB")
            if self.augment:
                img = self.color_jitter(img)
            if self.transform:
                img = self.transform(img)
            panels.append(img)
        
        panels = torch.stack(panels, dim=0)  # [4, 3, 224, 224]
        label_seq = rec["label_seq"]
        lang = 0 if rec["lang"] == "zh" else 1
        
        result = {
            "panels": panels,
            "lang": lang,
            "sample_name": rec["sample_name"],
        }
        
        if label_seq:
            label_idx = PERM_TO_IDX[tuple(label_seq)]
            result["label_idx"] = torch.tensor(label_idx, dtype=torch.long)
            result["label_seq"] = label_seq
        else:
            result["label_idx"] = torch.tensor(0, dtype=torch.long)
            result["label_seq"] = [0, 1, 2, 3]
        
        return result


class End2EndOrderModel(nn.Module):
    def __init__(self, num_classes=22, freeze_vit_blocks=8):
        super().__init__()
        self.vit = torchvision.models.vit_b_16(weights="DEFAULT")
        
        # Freeze specified number of ViT encoder blocks
        for block in self.vit.encoder.layers[:freeze_vit_blocks]:
            for param in block.parameters():
                param.requires_grad = False
        
        feat_dim = self.vit.hidden_dim  # 768
        
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, panels):
        # panels: [B, 4, 3, 224, 224]
        B = panels.size(0)
        panels_flat = panels.view(B * 4, 3, 224, 224)
        
        # ViT forward
        x = self.vit._process_input(panels_flat)  # [B*4, 768, 14, 14]
        n = x.shape[0]
        
        # Expand class token
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)
        cls_tokens = x[:, 0]  # [B*4, 768]
        
        cls_tokens = cls_tokens.view(B, 4, -1)
        cls_tokens = cls_tokens.flatten(1)  # [B, 4*768]
        
        logits = self.classifier(cls_tokens)  # [B, 22]
        return logits


def compute_exact_match(pred_seqs, gt_seqs):
    correct = sum(1 for p, g in zip(pred_seqs, gt_seqs) if p == g)
    return correct / len(gt_seqs) if gt_seqs else 0.0


def compute_macro_f1(pred_seqs, gt_seqs):
    if not gt_seqs:
        return 0.0
    pos_correct = [0, 0, 0, 0]
    pos_total = [0, 0, 0, 0]
    for p, g in zip(pred_seqs, gt_seqs):
        for i in range(4):
            pos_total[i] += 1
            if p[i] == g[i]:
                pos_correct[i] += 1
    pos_acc = [pos_correct[i] / pos_total[i] if pos_total[i] > 0 else 0.0 for i in range(4)]
    return sum(pos_acc) / 4.0, pos_acc


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_gts = []
    all_langs = []
    
    with torch.no_grad():
        for batch in dataloader:
            panels = batch["panels"].to(device)
            label_idx = batch["label_idx"].to(device)
            langs = batch["lang"].tolist()
            
            logits = model(panels)
            pred_idx = logits.argmax(dim=1).cpu().tolist()
            
            for pi, li, lang in zip(pred_idx, label_idx.cpu().tolist(), langs):
                pred_seq = IDX_TO_PERM[pi]
                gt_seq = IDX_TO_PERM[li]
                all_preds.append(pred_seq)
                all_gts.append(gt_seq)
                all_langs.append(lang)
    
    # Overall
    overall_em = compute_exact_match(all_preds, all_gts)
    overall_f1, overall_pos = compute_macro_f1(all_preds, all_gts)
    
    # By language
    zh_preds = [p for p, l in zip(all_preds, all_langs) if l == 0]
    zh_gts = [g for g, l in zip(all_gts, all_langs) if l == 0]
    en_preds = [p for p, l in zip(all_preds, all_langs) if l == 1]
    en_gts = [g for g, l in zip(all_gts, all_langs) if l == 1]
    
    zh_em = compute_exact_match(zh_preds, zh_gts) if zh_gts else 0.0
    zh_f1, zh_pos = compute_macro_f1(zh_preds, zh_gts) if zh_gts else (0.0, [0.0]*4)
    en_em = compute_exact_match(en_preds, en_gts) if en_gts else 0.0
    en_f1, en_pos = compute_macro_f1(en_preds, en_gts) if en_gts else (0.0, [0.0]*4)
    
    return {
        "overall": {"exact_match": overall_em, "macro_f1": overall_f1, "pos_acc": overall_pos},
        "zh": {"exact_match": zh_em, "macro_f1": zh_f1, "pos_acc": zh_pos},
        "en": {"exact_match": en_em, "macro_f1": en_f1, "pos_acc": en_pos},
    }


def main():
    print(f"Device: {DEVICE}")
    
    # Prepare data
    print("Loading metadata...")
    train_records = load_metadata("train")
    valid_records = load_metadata("valid")
    
    print(f"Train: {len(train_records)}, Valid: {len(valid_records)}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = ComicPanelDataset(train_records, transform=transform, augment=True)
    valid_dataset = ComicPanelDataset(valid_records, transform=transform, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model
    model = End2EndOrderModel(num_classes=22, freeze_vit_blocks=8).to(DEVICE)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total:,} total, {trainable:,} trainable ({100*trainable/total:.1f}%)")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    
    best_val_em = -1.0
    patience_counter = 0
    best_epoch = -1
    history = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            panels = batch["panels"].to(DEVICE)
            labels = batch["label_idx"].to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(panels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation (only on valid to save time; full eval done after training)
        val_metrics = evaluate(model, valid_loader, DEVICE)
        
        val_em = val_metrics["overall"]["exact_match"]
        scheduler.step(val_em)
        
        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_em": None,
            "val_em": val_em,
            "val_f1": val_metrics["overall"]["macro_f1"],
        })
        
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
              f"val_em={val_em:.4f} (ZH={val_metrics['zh']['exact_match']:.4f}, EN={val_metrics['en']['exact_match']:.4f}), "
              f"val_f1={val_metrics['overall']['macro_f1']:.4f}")
        
        if val_em > best_val_em:
            best_val_em = val_em
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(f"  -> New best model saved (val_em={val_em:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{PATIENCE})")
        
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Save history
    with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"Training complete. Best val_em={best_val_em:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
