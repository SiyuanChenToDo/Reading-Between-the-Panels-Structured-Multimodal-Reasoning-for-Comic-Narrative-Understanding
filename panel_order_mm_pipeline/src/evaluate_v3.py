"""
Evaluate v3 model.
"""
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from train_v3 import ComicOrderModel, ComicDataset, collate_fn


def compute_macro_f1(predictions, targets):
    def parse_order(order_str):
        try:
            return list(map(int, order_str.split()))
        except:
            return []
    position_correct = [0, 0, 0, 0]
    position_total = [0, 0, 0, 0]
    full_correct = 0
    for pred, target in zip(predictions, targets):
        pred_list = parse_order(pred.strip())
        target_list = parse_order(target.strip())
        if len(pred_list) != 4 or len(target_list) != 4:
            continue
        if pred_list == target_list:
            full_correct += 1
        for i in range(4):
            position_total[i] += 1
            if pred_list[i] == target_list[i]:
                position_correct[i] += 1
    position_f1 = []
    for i in range(4):
        position_f1.append(position_correct[i] / position_total[i] if position_total[i] > 0 else 0.0)
    macro_f1 = sum(position_f1) / 4 if position_f1 else 0.0
    full_accuracy = full_correct / len(predictions) if predictions else 0.0
    return macro_f1, full_accuracy, position_f1, full_correct


def evaluate_split(model, dataloader, device):
    model.eval()
    all_preds, all_gts, all_langs = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            token_ids = batch["token_ids"].to(device)
            panel_feats = batch["panel_feats"].to(device)
            lang_id = batch["lang_id"].to(device)
            B = panel_feats.size(0)
            spatial_pos = torch.arange(4, device=device).unsqueeze(0).expand(B, -1)
            scores = model(token_ids, panel_feats, lang_id, spatial_pos)
            preds = torch.argsort(scores, dim=1, descending=True).cpu().numpy()
            all_preds.extend([" ".join(map(str, p)) for p in preds.tolist()])
            all_gts.extend(batch["label_str"])
            all_langs.extend(["zh" if lid == 1 else "en" if lid == 2 else "unk" for lid in batch["lang_id"].tolist()])
    return all_preds, all_gts, all_langs


def compute_by_lang(all_preds, all_gts, all_langs):
    results = {}
    for lang in ["zh", "en"]:
        lp = [p for p, l in zip(all_preds, all_langs) if l == lang]
        lg = [t for t, l in zip(all_gts, all_langs) if l == lang]
        if lp:
            macro_f1, acc, pos_f1, fc = compute_macro_f1(lp, lg)
            results[lang] = {
                "macro_f1": macro_f1, "accuracy": acc,
                "position_f1": pos_f1, "full_correct": fc,
                "total_samples": len(lp),
            }
    if all_preds:
        macro_f1, acc, pos_f1, fc = compute_macro_f1(all_preds, all_gts)
        results["overall"] = {
            "macro_f1": macro_f1, "accuracy": acc,
            "position_f1": pos_f1, "full_correct": fc,
            "total_samples": len(all_preds),
        }
    return results


def print_results(split_name, lang_results):
    print(f"\n{'='*60}")
    print(f"【{split_name}】")
    print(f"{'='*60}")
    for lang in ["zh", "en", "overall"]:
        if lang not in lang_results:
            continue
        r = lang_results[lang]
        label = "整体 OVERALL" if lang == "overall" else lang.upper()
        print(f"\n  [{label}]")
        print(f"    样本数: {r['total_samples']}")
        print(f"    Macro-F1: {r['macro_f1']:.4f} ({r['macro_f1']*100:.2f}%)")
        print(f"    完整准确率: {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%)")
        for i, f1 in enumerate(r['position_f1']):
            print(f"      位置 {i}: {f1:.4f} ({f1*100:.2f}%)")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../data")
    feat_dir = os.path.join(script_dir, "../data_split/features")
    output_dir = os.path.join(script_dir, "../outputs/mm-panel-order-v3")
    
    with open(os.path.join(data_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    dataset = load_from_disk(os.path.join(data_dir, "dataset"))
    train_ds = ComicDataset(dataset["train"], vocab, os.path.join(feat_dir, "train"), augment=False)
    val_ds = ComicDataset(dataset["validation"], vocab, os.path.join(feat_dir, "valid"), augment=False)
    test_ds = ComicDataset(dataset["test"], vocab, os.path.join(feat_dir, "test"), augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComicOrderModel(vocab_size=len(vocab), num_layers=3, dropout=0.3).to(device)
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt"), map_location=device))
    
    output_json = {}
    
    train_preds, train_gts, train_langs = evaluate_split(model, train_loader, device)
    train_results = compute_by_lang(train_preds, train_gts, train_langs)
    print_results("训练集 Train", train_results)
    output_json["train"] = {k: {kk: vv for kk, vv in v.items()} for k, v in train_results.items()}
    
    val_preds, val_gts, val_langs = evaluate_split(model, val_loader, device)
    val_results = compute_by_lang(val_preds, val_gts, val_langs)
    print_results("验证集 Valid", val_results)
    output_json["valid"] = {k: {kk: vv for kk, vv in v.items()} for k, v in val_results.items()}
    
    test_preds, test_langs = evaluate_split(model, test_loader, device)[:2], []
    # Re-run test inference properly
    test_preds, test_langs = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            token_ids = batch["token_ids"].to(device)
            panel_feats = batch["panel_feats"].to(device)
            lang_id = batch["lang_id"].to(device)
            B = panel_feats.size(0)
            spatial_pos = torch.arange(4, device=device).unsqueeze(0).expand(B, -1)
            scores = model(token_ids, panel_feats, lang_id, spatial_pos)
            preds = torch.argsort(scores, dim=1, descending=True).cpu().numpy()
            test_preds.extend([" ".join(map(str, p)) for p in preds.tolist()])
            test_langs.extend(["zh" if lid == 1 else "en" if lid == 2 else "unk" for lid in batch["lang_id"].tolist()])
    
    output_json["test"] = {"predictions": test_preds, "langs": test_langs}
    
    out_path = os.path.join(script_dir, "../outputs/evaluation_results_v3.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
