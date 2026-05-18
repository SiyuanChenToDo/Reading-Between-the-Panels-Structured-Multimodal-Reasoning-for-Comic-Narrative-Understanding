"""
Train a multimodal temporal ordering model v3.
Key fixes:
- Shuffle panel order during training
- Spatial + language embeddings
- Stepwise masked cross-entropy loss (stronger signal than ListMLE)
- Oversample Chinese training data
- Higher dropout and regularization
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import load_from_disk
from tqdm import tqdm


def tokenize(text, vocab, max_len=64):
    words = text.lower().split()[:max_len]
    ids = [vocab.get(w, vocab["<unk>"]) for w in words]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, output_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, token_ids):
        x = self.embedding(token_ids)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.proj(h)


class ComicOrderModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=8, num_layers=3, dim_feedforward=2048, dropout=0.3):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, output_dim=d_model)
        self.lang_embed = nn.Embedding(3, d_model)
        self.spatial_embed = nn.Embedding(4, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.score_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, token_ids, panel_feats, lang_ids=None, spatial_positions=None):
        B = panel_feats.size(0)
        text_feat = self.text_encoder(token_ids).unsqueeze(1)
        
        if lang_ids is not None:
            text_feat = text_feat + self.lang_embed(lang_ids).unsqueeze(1)
        
        panel_feats = panel_feats.clone()
        if spatial_positions is not None:
            panel_feats = panel_feats + self.spatial_embed(spatial_positions)
        
        x = torch.cat([text_feat, panel_feats], dim=1)
        x = self.transformer(x)
        scores = self.score_head(x[:, 1:, :]).squeeze(-1)
        return scores


def stepwise_masked_ce_loss(scores, target_perm):
    """
    scores: [B, 4] logits for each panel
    target_perm: [B, 4] ordered panel indices
    """
    B = scores.size(0)
    device = scores.device
    loss = 0.0
    
    for step in range(4):
        target = target_perm[:, step]  # [B]
        loss += F.cross_entropy(scores, target)
        
        # Mask out selected panels for subsequent steps
        mask = torch.ones_like(scores, dtype=torch.bool)
        for b in range(B):
            mask[b, target[b]] = False
        scores = scores.masked_fill(~mask, -1e9)
    
    return loss / 4.0


class ComicDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, vocab, feat_dir, augment=False):
        self.dataset = hf_dataset
        self.vocab = vocab
        self.feat_dir = feat_dir
        self.augment = augment
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        token_ids = tokenize(item["prompt"], self.vocab)
        sample_name = item["sample_name"]
        feat_path = os.path.join(self.feat_dir, f"{sample_name}.npy")
        panel_feats = np.load(feat_path).astype(np.float32)
        
        if self.augment:
            panel_feats += np.random.randn(*panel_feats.shape).astype(np.float32) * 0.1
        
        label_seq = item["label_seq"]
        lang = item.get("lang", "unknown")
        lang_id = 1 if lang == "zh" else 2 if lang == "en" else 0
        
        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "panel_feats": torch.tensor(panel_feats),
            "label_seq": torch.tensor(label_seq, dtype=torch.long),
            "label_str": item["label_str"],
            "lang_id": lang_id,
        }


def collate_fn(batch):
    return {
        "token_ids": torch.stack([b["token_ids"] for b in batch]),
        "panel_feats": torch.stack([b["panel_feats"] for b in batch]),
        "label_seq": torch.stack([b["label_seq"] for b in batch]),
        "label_str": [b["label_str"] for b in batch],
        "lang_id": torch.tensor([b["lang_id"] for b in batch], dtype=torch.long),
    }


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_gts = []
    with torch.no_grad():
        for batch in dataloader:
            token_ids = batch["token_ids"].to(device)
            panel_feats = batch["panel_feats"].to(device)
            lang_id = batch["lang_id"].to(device)
            B = panel_feats.size(0)
            spatial_pos = torch.arange(4, device=device).unsqueeze(0).expand(B, -1)
            scores = model(token_ids, panel_feats, lang_id, spatial_pos)
            preds = torch.argsort(scores, dim=1, descending=True).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_gts.extend(batch["label_seq"].numpy().tolist())
    
    exact_match = sum(np.array(p).tolist() == np.array(g).tolist() for p, g in zip(all_preds, all_gts)) / len(all_preds)
    return float(exact_match)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../data")
    feat_dir = os.path.join(script_dir, "../data_split/features")
    output_dir = os.path.join(script_dir, "../outputs/mm-panel-order-v3")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(data_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    dataset = load_from_disk(os.path.join(data_dir, "dataset"))
    train_ds = ComicDataset(dataset["train"], vocab, os.path.join(feat_dir, "train"), augment=True)
    val_ds = ComicDataset(dataset["validation"], vocab, os.path.join(feat_dir, "valid"), augment=False)
    
    # Create oversampled Chinese data: repeat zh indices 4x to balance with en
    train_indices = list(range(len(train_ds)))
    train_langs = dataset["train"]["lang"]
    zh_indices = [i for i, l in enumerate(train_langs) if l == "zh"]
    en_indices = [i for i, l in enumerate(train_langs) if l == "en"]
    
    # Oversample zh 4x to get ~384*4 = 1536 zh samples, similar to en
    oversampled_indices = en_indices + zh_indices * 4
    # Shuffle
    np.random.seed(42)
    np.random.shuffle(oversampled_indices)
    
    sampler = torch.utils.data.SubsetRandomSampler(oversampled_indices)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComicOrderModel(vocab_size=len(vocab), num_layers=3, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    
    best_acc = -1.0
    patience_counter = 0
    
    for epoch in range(1, 101):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            token_ids = batch["token_ids"].to(device)
            panel_feats = batch["panel_feats"].to(device)
            label_seq = batch["label_seq"].to(device)
            lang_id = batch["lang_id"].to(device)
            B = panel_feats.size(0)
            
            # Shuffle panel order during training
            perm = torch.randperm(4, device=device)
            shuffled_panels = panel_feats[:, perm, :]
            spatial_pos = perm.unsqueeze(0).expand(B, -1)
            
            scores_shuffled = model(token_ids, shuffled_panels, lang_id, spatial_pos)
            
            # Unshuffle scores
            unshuffle_idx = torch.zeros_like(perm)
            unshuffle_idx[perm] = torch.arange(4, device=device)
            scores = scores_shuffled[:, unshuffle_idx]
            
            loss = stepwise_masked_ce_loss(scores, label_seq)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        val_acc = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_exact_match={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"  -> Saved best model (acc={best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print("Early stopping triggered.")
                break
    
    print(f"Training finished. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
