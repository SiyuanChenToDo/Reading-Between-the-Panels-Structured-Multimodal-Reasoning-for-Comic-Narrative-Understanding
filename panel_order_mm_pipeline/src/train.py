"""
Train a multimodal temporal ordering model with:
- Frozen ViT features for 4 panels
- Learnable text encoder for prompts
- Transformer fusion + ListMLE loss
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
        # token_ids: [B, L]
        x = self.embedding(token_ids)  # [B, L, E]
        _, (h_n, _) = self.lstm(x)  # h_n: [2, B, H]
        h = torch.cat([h_n[0], h_n[1]], dim=-1)  # [B, 2H]
        return self.proj(h)  # [B, output_dim]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ComicOrderModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=8, num_layers=2, dim_feedforward=2048):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, output_dim=d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=16)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_head = nn.Linear(d_model, 1)
    
    def forward(self, token_ids, panel_feats):
        # token_ids: [B, L]
        # panel_feats: [B, 4, 768]
        B = panel_feats.size(0)
        text_feat = self.text_encoder(token_ids).unsqueeze(1)  # [B, 1, 768]
        x = torch.cat([text_feat, panel_feats], dim=1)  # [B, 5, 768]
        x = self.pos_encoder(x)
        x = self.transformer(x)  # [B, 5, 768]
        scores = self.score_head(x[:, 1:, :]).squeeze(-1)  # [B, 4]
        return scores


def list_mle_loss(scores, target_perm):
    """
    scores: [B, 4] logits for panel 0,1,2,3
    target_perm: [B, 4] ordered indices, e.g. [[1,0,2,3], ...]
    """
    loss = 0.0
    for i in range(4):
        remaining = target_perm[:, i:]  # [B, 4-i]
        remaining_scores = torch.gather(scores, 1, remaining)  # [B, 4-i]
        correct_score = remaining_scores[:, 0]
        loss += (torch.logsumexp(remaining_scores, dim=1) - correct_score).mean()
    return loss / 4.0


class ComicDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, vocab, feat_dir):
        self.dataset = hf_dataset
        self.vocab = vocab
        self.feat_dir = feat_dir
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        token_ids = tokenize(item["prompt"], self.vocab)
        sample_name = item["sample_name"]
        feat_path = os.path.join(self.feat_dir, f"{sample_name}.npy")
        panel_feats = np.load(feat_path).astype(np.float32)  # [4, 768]
        label_seq = item["label_seq"]
        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "panel_feats": torch.tensor(panel_feats),
            "label_seq": torch.tensor(label_seq, dtype=torch.long),
            "label_str": item["label_str"],
            "lang": item.get("lang", "unknown"),
        }


def collate_fn(batch):
    return {
        "token_ids": torch.stack([b["token_ids"] for b in batch]),
        "panel_feats": torch.stack([b["panel_feats"] for b in batch]),
        "label_seq": torch.stack([b["label_seq"] for b in batch]),
        "label_str": [b["label_str"] for b in batch],
        "lang": [b["lang"] for b in batch],
    }


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_gts = []
    with torch.no_grad():
        for batch in dataloader:
            token_ids = batch["token_ids"].to(device)
            panel_feats = batch["panel_feats"].to(device)
            scores = model(token_ids, panel_feats)  # [B, 4]
            preds = torch.argsort(scores, dim=1, descending=True).cpu().numpy()  # [B, 4]
            all_preds.extend(preds.tolist())
            all_gts.extend(batch["label_seq"].numpy().tolist())
    
    exact_match = np.mean([np.array(p) == np.array(g) for p, g in zip(all_preds, all_gts)])
    return exact_match


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../data")
    feat_dir = os.path.join(script_dir, "../data_split/features")
    output_dir = os.path.join(script_dir, "../outputs/mm-panel-order")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(data_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    dataset = load_from_disk(os.path.join(data_dir, "dataset"))
    train_ds = ComicDataset(dataset["train"], vocab, os.path.join(feat_dir, "train"))
    val_ds = ComicDataset(dataset["validation"], vocab, os.path.join(feat_dir, "valid"))
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComicOrderModel(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    
    best_acc = -1.0
    patience_counter = 0
    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            token_ids = batch["token_ids"].to(device)
            panel_feats = batch["panel_feats"].to(device)
            label_seq = batch["label_seq"].to(device)
            
            scores = model(token_ids, panel_feats)
            loss = list_mle_loss(scores, label_seq)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        val_acc = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_exact_match={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"  -> Saved best model (acc={best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping triggered.")
                break
    
    print(f"Training finished. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
