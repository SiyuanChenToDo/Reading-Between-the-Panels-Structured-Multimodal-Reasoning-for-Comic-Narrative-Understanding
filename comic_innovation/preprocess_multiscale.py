"""
数据预处理：将 Task1 合成图切割为 4 个独立 panel，
并生成多尺度数据集 JSON（composite + 4 panels = 5 images/sample）。
同时生成多任务联合数据集（task1_multiscale + task2 混合）。
"""

import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def split_composite_image(img_path: str, output_dir: str) -> list[str]:
    """将 2x2 合成图切割为 4 个 panel。"""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    mid_w, mid_h = w // 2, h // 2

    stem = Path(img_path).stem
    suffix = Path(img_path).suffix
    os.makedirs(output_dir, exist_ok=True)

    panels = [
        img.crop((0, 0, mid_w, mid_h)),
        img.crop((mid_w, 0, w, mid_h)),
        img.crop((0, mid_h, mid_w, h)),
        img.crop((mid_w, mid_h, w, h)),
    ]

    paths = []
    for i, panel in enumerate(panels):
        fname = f"{stem}_panel{i}{suffix}"
        out_path = os.path.join(output_dir, fname)
        panel.save(out_path)
        paths.append(fname)
    return paths


def process_task1_images(data_root: str):
    """遍历 task1 图片目录，切割所有合成图。"""
    images_base = os.path.join(data_root, "task1", "train", "images")
    for lang_dir in ["zh", "en"]:
        src_dir = os.path.join(images_base, lang_dir)
        if not os.path.isdir(src_dir):
            continue
        out_dir = os.path.join(images_base, f"{lang_dir}_split")
        os.makedirs(out_dir, exist_ok=True)

        files = sorted(f for f in os.listdir(src_dir) if f.endswith((".jpg", ".png")))
        print(f"Splitting {len(files)} images in {src_dir} -> {out_dir}")
        for fname in tqdm(files, desc=f"task1/train/{lang_dir}"):
            split_composite_image(os.path.join(src_dir, fname), out_dir)

    for split in ["valid", "test"]:
        split_images = os.path.join(data_root, "task1", split, "images")
        for lang_dir in ["zh", "en"]:
            src_dir = os.path.join(split_images, lang_dir)
            if not os.path.isdir(src_dir):
                continue
            out_dir = os.path.join(split_images, f"{lang_dir}_split")
            os.makedirs(out_dir, exist_ok=True)
            files = sorted(f for f in os.listdir(src_dir) if f.endswith((".jpg", ".png")))
            print(f"Splitting {len(files)} images in {src_dir} -> {out_dir}")
            for fname in tqdm(files, desc=f"task1/{split}/{lang_dir}"):
                split_composite_image(os.path.join(src_dir, fname), out_dir)


MULTISCALE_PROMPT_EN = (
    "<image>\nThe above image shows the full 4-panel comic with shuffled order.\n"
    "Below are the individual panels for detailed analysis:\n"
    "<image> Panel at position 0 (top-left)\n"
    "<image> Panel at position 1 (top-right)\n"
    "<image> Panel at position 2 (bottom-left)\n"
    "<image> Panel at position 3 (bottom-right)\n\n"
    "Analyze both the overall layout and individual panel content to determine "
    "the correct reading order (use space-separated numbers, e.g., \"0 1 2 3\"). "
    "Only output the number sequence without any explanation."
)

MULTISCALE_PROMPT_ZH = (
    "<image>\n上图是顺序打乱的完整四格漫画。\n"
    "以下是各画格的单独图像，供详细分析：\n"
    "<image> 位置 0（左上）\n"
    "<image> 位置 1（右上）\n"
    "<image> 位置 2（左下）\n"
    "<image> 位置 3（右下）\n\n"
    "请结合整体布局和每个画格的内容，推断正确的阅读顺序（用空格分隔的数字表示，例如\"0 1 2 3\"）。"
    "仅输出数字序列，不要添加任何解释。"
)


def create_multiscale_json(src_json_path: str, out_json_path: str):
    """把单图 task1 JSON 转换为 5 图多尺度版本。"""
    with open(src_json_path, "r") as f:
        data = json.load(f)

    new_data = []
    for item in data:
        orig_img = item["images"][0]
        img_dir = str(Path(orig_img).parent)
        stem = Path(orig_img).stem
        suffix = Path(orig_img).suffix

        lang = "zh" if "/zh/" in orig_img else "en"
        prompt = MULTISCALE_PROMPT_ZH if lang == "zh" else MULTISCALE_PROMPT_EN
        split_dir = img_dir.replace(f"/{lang}", f"/{lang}_split")

        panel_paths = [f"{split_dir}/{stem}_panel{i}{suffix}" for i in range(4)]

        answer = item["conversations"][1]["value"]
        new_item = {
            "images": [orig_img] + panel_paths,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": answer},
            ],
        }
        new_data.append(new_item)

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"Created multiscale dataset: {out_json_path} ({len(new_data)} samples)")


def create_joint_json(task1_json: str, task2_json: str, out_json: str, zh_upsample: int = 2):
    """合并 task1(multiscale) 和 task2 数据用于多任务训练。
    对中文 task1 样本进行 upsample 以平衡语言比例。
    """
    with open(task1_json, "r") as f:
        t1 = json.load(f)
    with open(task2_json, "r") as f:
        t2 = json.load(f)

    t1_balanced = []
    for item in t1:
        is_zh = "/zh/" in item["images"][0]
        repeat = zh_upsample if is_zh else 1
        t1_balanced.extend([item] * repeat)

    merged = t1_balanced + t2
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Created joint dataset: {out_json} (task1={len(t1_balanced)}[orig {len(t1)}], task2={len(t2)}, total={len(merged)})")


def update_dataset_info(dataset_info_path: str):
    """在 dataset_info.json 中注册新数据集。"""
    with open(dataset_info_path, "r") as f:
        info = json.load(f)

    new_entries = {
        "ccac2025_task1_train_multiscale": {
            "file_name": "task1/train/train_multiscale.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
        },
        "ccac2025_task1_valid_multiscale": {
            "file_name": "task1/valid/valid_multiscale.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
        },
        "ccac2025_task1_test_multiscale": {
            "file_name": "task1/test/test_multiscale.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
        },
        "ccac2025_joint_train": {
            "file_name": "joint/train_joint.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
        },
    }

    info.update(new_entries)
    with open(dataset_info_path, "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"Updated dataset_info.json with {len(new_entries)} new entries")


def main():
    data_root = "/root/autodl-tmp/LlamaFactory/data/ccac2025_complete"

    print("=" * 60)
    print("Step 1: Splitting composite images into panels")
    print("=" * 60)
    process_task1_images(data_root)

    print("\n" + "=" * 60)
    print("Step 2: Creating multiscale dataset JSONs")
    print("=" * 60)
    for split in ["train", "valid", "test"]:
        src = os.path.join(data_root, "task1", split, f"{split}.json")
        dst = os.path.join(data_root, "task1", split, f"{split}_multiscale.json")
        if os.path.exists(src):
            create_multiscale_json(src, dst)

    print("\n" + "=" * 60)
    print("Step 3: Creating joint multi-task dataset")
    print("=" * 60)
    t1_ms = os.path.join(data_root, "task1", "train", "train_multiscale.json")
    t2 = os.path.join(data_root, "task2", "train", "train.json")
    joint = os.path.join(data_root, "joint", "train_joint.json")
    if os.path.exists(t1_ms) and os.path.exists(t2):
        create_joint_json(t1_ms, t2, joint)

    print("\n" + "=" * 60)
    print("Step 4: Updating dataset_info.json")
    print("=" * 60)
    update_dataset_info(os.path.join(data_root, "dataset_info.json"))

    print("\nDone!")


if __name__ == "__main__":
    main()
