"""
Task1 数据平衡脚本（样本复制版）

核心功能：
1. 排列级平衡采样：对稀有排列上采样，使训练集排列分布更均匀
2. 中文样本上采样：弥补中英文数量差距
3. 验证集保持原样，不做任何修改

注意：本脚本仅做样本复制（不修改图像），配合训练时的 dropout/weight_decay 缓解过拟合。
若需图像增强，建议在外部预处理流程中完成。
"""

import json
import os
import random
from pathlib import Path
from collections import Counter, defaultdict
from copy import deepcopy


def parse_order(item: dict) -> tuple:
    """从数据项解析排列。"""
    ans = item["conversations"][1]["value"].strip()
    nums = tuple(int(x) for x in ans.split() if x.isdigit())
    if len(nums) == 4 and sorted(nums) == [0, 1, 2, 3]:
        return nums
    return None


def get_lang(item: dict) -> str:
    """判断语言。"""
    img_path = item["images"][0]
    return "zh" if "/zh/" in img_path else "en"


def balance_train_dataset(
    data: list,
    target_per_perm: int = None,
    target_total: int = None,
) -> list:
    """
    对训练集进行排列级平衡上采样（仅复制样本，不修改图像）。

    Args:
        data: 原始训练数据列表
        target_per_perm: 每种排列的目标样本数
        target_total: 目标总样本数（优先）
    """
    perm_to_items = defaultdict(list)
    for item in data:
        order = parse_order(item)
        if order:
            perm_to_items[order].append(item)

    n_perms = len(perm_to_items)
    print(f"  原始数据: {len(data)} 条, {n_perms} 种排列")
    for perm, items in sorted(perm_to_items.items(), key=lambda x: -len(x[1])):
        print(f"    {perm}: {len(items)}")

    if target_total:
        target_per_perm = max(target_total // n_perms, max(len(v) for v in perm_to_items.values()))
    elif target_per_perm is None:
        max_count = max(len(v) for v in perm_to_items.values())
        target_per_perm = max(int(max_count * 0.9), 30)

    print(f"  目标: 每种排列至少 {target_per_perm} 条")

    balanced = []
    for perm, items in perm_to_items.items():
        n_needed = target_per_perm - len(items)
        balanced.extend(items)
        if n_needed > 0:
            # 随机采样（有放回）补充到目标数量
            extra = random.choices(items, k=n_needed)
            balanced.extend(deepcopy(x) for x in extra)

    random.shuffle(balanced)

    # 统计
    new_counts = Counter(parse_order(x) for x in balanced if parse_order(x))
    print(f"  平衡后: {len(balanced)} 条, {len(new_counts)} 种排列")
    return balanced


def create_balanced_datasets(
    data_root: str = "/root/autodl-tmp/LlamaFactory/data/ccac2025_complete",
    zh_target_total: int = 1500,
    en_target_total: int = 1500,
):
    """创建平衡后的训练集，验证集保持原样。"""
    data_root = Path(data_root)

    # ---- 处理训练集 ----
    print(f"\n{'='*60}")
    print("Processing TRAIN (balancing)")
    print(f"{'='*60}")

    train_json = data_root / "task1" / "train" / "train.json"
    with open(train_json) as f:
        train_data = json.load(f)

    zh_train = [x for x in train_data if get_lang(x) == "zh"]
    en_train = [x for x in train_data if get_lang(x) == "en"]

    print(f"\n  [中文] 原始 {len(zh_train)} 条")
    zh_balanced = balance_train_dataset(zh_train, target_total=zh_target_total)

    print(f"\n  [英文] 原始 {len(en_train)} 条")
    en_balanced = balance_train_dataset(en_train, target_total=en_target_total)

    merged_train = zh_balanced + en_balanced
    random.shuffle(merged_train)

    # 保存平衡后的单图训练集
    out_train = data_root / "task1" / "train" / "train_balanced.json"
    with open(out_train, "w") as f:
        json.dump(merged_train, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_train} ({len(merged_train)} samples)")

    # 生成多尺度版本
    from preprocess_multiscale import create_multiscale_json
    out_train_ms = data_root / "task1" / "train" / "train_multiscale_balanced.json"
    create_multiscale_json(str(out_train), str(out_train_ms))

    # ---- 验证集：保持原样，只确保 multiscale 版本存在 ----
    print(f"\n{'='*60}")
    print("Processing VALID (keep original)")
    print(f"{'='*60}")

    valid_json = data_root / "task1" / "valid" / "valid.json"
    valid_ms_json = data_root / "task1" / "valid" / "valid_multiscale.json"

    if valid_ms_json.exists():
        print(f"  Valid multiscale already exists: {valid_ms_json}")
    else:
        create_multiscale_json(str(valid_json), str(valid_ms_json))
        print(f"  Created: {valid_ms_json}")

    # ---- 更新 dataset_info ----
    update_balanced_dataset_info(str(data_root / "dataset_info.json"))


def update_balanced_dataset_info(dataset_info_path: str):
    """注册平衡后的训练集到 dataset_info.json。验证集使用原有的 multiscale。"""
    with open(dataset_info_path, "r") as f:
        info = json.load(f)

    new_entries = {
        "ccac2025_task1_train_multiscale_balanced": {
            "file_name": "task1/train/train_multiscale_balanced.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
        },
        # 验证集仍使用原有未修改的 multiscale
        "ccac2025_task1_valid_multiscale_balanced": {
            "file_name": "task1/valid/valid_multiscale.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
        },
    }

    info.update(new_entries)
    with open(dataset_info_path, "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"\nUpdated dataset_info.json")


if __name__ == "__main__":
    create_balanced_datasets(
        zh_target_total=1500,
        en_target_total=1500,
    )
