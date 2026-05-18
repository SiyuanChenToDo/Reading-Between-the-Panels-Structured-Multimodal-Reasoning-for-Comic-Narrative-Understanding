"""
为中文训练集生成缺失排列的合成样本。

核心方法：
1. 识别训练集中缺失的排列（当前缺失4种）
2. 对每种缺失排列，选取现有样本的4个panel图像
3. 按照目标排列重新组合成新的composite图
4. 切割新的composite图为4个panel，保存到专用目录
5. 生成对应的多尺度JSON条目

这样可以在不污染验证集的前提下，让训练集覆盖全部24种排列。
"""

import json
import os
import random
from pathlib import Path
from PIL import Image
from copy import deepcopy
from itertools import permutations

DATA_ROOT = Path("/root/autodl-tmp/LlamaFactory/data/ccac2025_complete")

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


def parse_order(item: dict) -> tuple:
    ans = item["conversations"][1]["value"].strip()
    nums = tuple(int(x) for x in ans.split() if x.isdigit())
    if len(nums) == 4 and sorted(nums) == [0, 1, 2, 3]:
        return nums
    return None


def get_lang(item: dict) -> str:
    return "zh" if "/zh/" in item["images"][0] else "en"


def load_panels_from_item(item: dict) -> list[Image.Image]:
    """从单图数据项加载composite并切割为4个panel。"""
    composite_path = DATA_ROOT / item["images"][0]
    composite = Image.open(composite_path).convert("RGB")
    w, h = composite.size
    mid_w, mid_h = w // 2, h // 2
    panels = [
        composite.crop((0, 0, mid_w, mid_h)),
        composite.crop((mid_w, 0, w, mid_h)),
        composite.crop((0, mid_h, mid_w, h)),
        composite.crop((mid_w, mid_h, w, h)),
    ]
    return panels


def create_composite_from_panels(panels: list[Image.Image], perm: tuple) -> Image.Image:
    """
    按照排列 perm 重新组合4个panel成新的composite图。
    perm[i] = 应该在位置 i 上放的原始panel索引。
    """
    assert len(panels) == 4 and len(perm) == 4
    w, h = panels[0].size
    composite = Image.new("RGB", (w * 2, h * 2))
    # 位置布局：0=左上, 1=右上, 2=左下, 3=右下
    positions = [(0, 0), (w, 0), (0, h), (w, h)]
    for pos_idx, panel_idx in enumerate(perm):
        x, y = positions[pos_idx]
        composite.paste(panels[panel_idx], (x, y))
    return composite


def split_composite_image(img: Image.Image, stem: str, suffix: str, output_dir: Path) -> list[str]:
    """切割composite图为4个panel。"""
    w, h = img.size
    mid_w, mid_h = w // 2, h // 2
    panels = [
        img.crop((0, 0, mid_w, mid_h)),
        img.crop((mid_w, 0, w, mid_h)),
        img.crop((0, mid_h, mid_w, h)),
        img.crop((mid_w, mid_h, w, h)),
    ]
    paths = []
    for i, panel in enumerate(panels):
        fname = f"{stem}_panel{i}{suffix}"
        out_path = output_dir / fname
        panel.save(out_path)
        paths.append(str(out_path))
    return paths


def generate_missing_permutations(target_count_per_perm: int = 50):
    """生成缺失排列的合成样本。"""
    # 加载原始训练数据
    with open(DATA_ROOT / "task1" / "train" / "train.json") as f:
        raw_data = json.load(f)

    zh_data = [x for x in raw_data if get_lang(x) == "zh"]
    all_perms = set(permutations([0, 1, 2, 3]))
    found_perms = {parse_order(x) for x in zh_data if parse_order(x)}
    missing_perms = sorted(all_perms - found_perms)

    print(f"中文训练集缺失排列: {missing_perms} (共 {len(missing_perms)} 种)")

    if not missing_perms:
        print("没有缺失排列，无需生成。")
        return

    # 创建输出目录
    synth_img_dir = DATA_ROOT / "task1" / "train" / "images" / "zh_synth"
    synth_split_dir = DATA_ROOT / "task1" / "train" / "images" / "zh_synth_split"
    synth_img_dir.mkdir(parents=True, exist_ok=True)
    synth_split_dir.mkdir(parents=True, exist_ok=True)

    synth_items = []
    counter = 0

    for perm in missing_perms:
        print(f"\n生成排列 {perm} 的合成样本...")
        for _ in range(target_count_per_perm):
            # 随机选一个现有中文样本
            src_item = random.choice(zh_data)
            panels = load_panels_from_item(src_item)

            # 按 perm 重新组合composite
            composite = create_composite_from_panels(panels, perm)

            # 保存composite
            stem = f"synth_{perm[0]}{perm[1]}{perm[2]}{perm[3]}_{counter:04d}"
            suffix = ".jpg"
            composite_path = synth_img_dir / f"{stem}{suffix}"
            composite.save(composite_path, quality=95)

            # 切割panel
            panel_paths = split_composite_image(composite, stem, suffix, synth_split_dir)

            # 构建多尺度数据项
            # 注意：panel_paths 已经是绝对/相对路径，需要转为相对 DATA_ROOT 的路径
            rel_composite = str(composite_path.relative_to(DATA_ROOT))
            rel_panels = [str(Path(p).relative_to(DATA_ROOT)) for p in panel_paths]

            answer = " ".join(str(x) for x in perm)
            new_item = {
                "images": [rel_composite] + rel_panels,
                "conversations": [
                    {"from": "human", "value": MULTISCALE_PROMPT_ZH},
                    {"from": "gpt", "value": answer},
                ],
            }
            synth_items.append(new_item)
            counter += 1

    print(f"\n共生成 {len(synth_items)} 个合成样本")

    # 加载已平衡的训练集
    with open(DATA_ROOT / "task1" / "train" / "train_balanced.json") as f:
        balanced_data = json.load(f)

    # 合并
    merged = balanced_data + synth_items
    random.shuffle(merged)

    # 保存
    out_json = DATA_ROOT / "task1" / "train" / "train_balanced_full.json"
    with open(out_json, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_json} ({len(merged)} samples)")

    # 手动生成多尺度版本（避免 create_multiscale_json 的路径替换问题）
    out_ms = DATA_ROOT / "task1" / "train" / "train_multiscale_balanced_full.json"
    ms_data = []
    for item in merged:
        orig_img = item["images"][0]
        img_dir = str(Path(orig_img).parent)
        stem = Path(orig_img).stem
        suffix = Path(orig_img).suffix

        # 判断语言
        if "/zh_synth/" in orig_img:
            prompt = MULTISCALE_PROMPT_ZH
            split_dir = img_dir.replace("/zh_synth", "/zh_synth_split")
        elif "/zh/" in orig_img:
            prompt = MULTISCALE_PROMPT_ZH
            split_dir = img_dir.replace("/zh", "/zh_split")
        else:
            from preprocess_multiscale import MULTISCALE_PROMPT_EN
            prompt = MULTISCALE_PROMPT_EN
            split_dir = img_dir.replace("/en", "/en_split")

        panel_paths = [f"{split_dir}/{stem}_panel{i}{suffix}" for i in range(4)]

        ms_item = {
            "images": [orig_img] + panel_paths,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": item["conversations"][1]["value"]},
            ],
        }
        ms_data.append(ms_item)

    with open(out_ms, "w") as f:
        json.dump(ms_data, f, ensure_ascii=False, indent=2)
    print(f"Created multiscale: {out_ms} ({len(ms_data)} samples)")

    # 更新 dataset_info
    update_dataset_info()


def update_dataset_info():
    info_path = DATA_ROOT / "dataset_info.json"
    with open(info_path) as f:
        info = json.load(f)

    info["ccac2025_task1_train_multiscale_balanced_full"] = {
        "file_name": "task1/train/train_multiscale_balanced_full.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"},
    }

    with open(info_path, "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("Updated dataset_info.json with balanced_full entry")


if __name__ == "__main__":
    random.seed(42)
    generate_missing_permutations(target_count_per_perm=50)
