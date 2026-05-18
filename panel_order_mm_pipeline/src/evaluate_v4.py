"""
Evaluate v4 end-to-end ViT model on train/valid/test.
"""

import os
import sys
import json

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_v4 import ComicPanelDataset, load_metadata, End2EndOrderModel, evaluate, IDX_TO_PERM

DATA_ROOT = "./data_e2e"
OUTPUT_DIR = "./outputs/mm-panel-order-v4"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = End2EndOrderModel(num_classes=22, freeze_vit_blocks=8).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"), map_location=DEVICE))

    results = {}

    for split in ["train", "valid", "test"]:
        records = load_metadata(split)
        dataset = ComicPanelDataset(records, transform=transform, augment=False)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

        if split == "test":
            # Test has no labels
            model.eval()
            all_preds = []
            all_names = []
            with torch.no_grad():
                for batch in loader:
                    panels = batch["panels"].to(DEVICE)
                    logits = model(panels)
                    pred_idx = logits.argmax(dim=1).cpu().tolist()
                    for pi, name in zip(pred_idx, batch["sample_name"]):
                        all_preds.append(IDX_TO_PERM[pi])
                        all_names.append(name)
            results["test"] = {
                "predictions": [{"sample_name": n, "pred_order": " ".join(map(str, p))} for n, p in zip(all_names, all_preds)]
            }
        else:
            metrics = evaluate(model, loader, DEVICE)
            results[split] = {
                "overall": {
                    "macro_f1": metrics["overall"]["macro_f1"],
                    "exact_match": metrics["overall"]["exact_match"],
                    "position_acc": metrics["overall"]["pos_acc"],
                },
                "zh": {
                    "macro_f1": metrics["zh"]["macro_f1"],
                    "exact_match": metrics["zh"]["exact_match"],
                    "position_acc": metrics["zh"]["pos_acc"],
                },
                "en": {
                    "macro_f1": metrics["en"]["macro_f1"],
                    "exact_match": metrics["en"]["exact_match"],
                    "position_acc": metrics["en"]["pos_acc"],
                },
            }
            print(f"\n[{split.upper()}]")
            print(f"  Overall EM: {metrics['overall']['exact_match']:.4f}, F1: {metrics['overall']['macro_f1']:.4f}")
            print(f"  ZH EM: {metrics['zh']['exact_match']:.4f}, F1: {metrics['zh']['macro_f1']:.4f}")
            print(f"  EN EM: {metrics['en']['exact_match']:.4f}, F1: {metrics['en']['macro_f1']:.4f}")

    out_path = os.path.join(OUTPUT_DIR, "evaluation_results_v4.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
