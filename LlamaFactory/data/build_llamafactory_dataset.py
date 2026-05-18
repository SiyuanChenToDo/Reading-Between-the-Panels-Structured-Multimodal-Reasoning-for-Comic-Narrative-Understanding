#!/usr/bin/env python3
"""
构建适配LlamaFactory的多模态数据集 - 完整版
包含训练集、验证集、测试集

用于CCAC2025四格漫画理解任务
"""

import json
import csv
import os
import shutil
import ast
from pathlib import Path

# 数据集根目录
DATASET_ROOT = "/root/autodl-tmp/dataset"
LLAMA_FACTORY_DATA = "/root/autodl-tmp/LlamaFactory/data"
OUTPUT_DIR = f"{LLAMA_FACTORY_DATA}/ccac2025_complete"

def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)

def copy_images(src_dir, dst_dir):
    """复制图片到输出目录"""
    if not os.path.exists(dst_dir):
        ensure_dir(dst_dir)
    
    count = 0
    for f in os.listdir(src_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
            src_path = os.path.join(src_dir, f)
            dst_path = os.path.join(dst_dir, f)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
            count += 1
    return count

def copy_all_images():
    """复制所有图片到输出目录"""
    print("=" * 70)
    print("复制图片到数据集目录")
    print("=" * 70)
    
    # 训练集图片
    print("\n【训练集】")
    train_zh_src = f"{DATASET_ROOT}/CCAC2025manga_train/CCAC2025manga_train/train_zh"
    train_en_src = f"{DATASET_ROOT}/CCAC2025manga_train/CCAC2025manga_train/train_en"
    train_zh_dst = f"{OUTPUT_DIR}/train_zh"
    train_en_dst = f"{OUTPUT_DIR}/train_en"
    
    count_zh = copy_images(train_zh_src, train_zh_dst)
    count_en = copy_images(train_en_src, train_en_dst)
    print(f"  中文图片: {count_zh} 张 -> {train_zh_dst}")
    print(f"  英文图片: {count_en} 张 -> {train_en_dst}")
    
    # 验证集图片
    print("\n【验证集】")
    valid_zh_task1_src = f"{DATASET_ROOT}/CCAC2025_valid/CCAC2025_valid/valid_zh_task1"
    valid_zh_task2_src = f"{DATASET_ROOT}/CCAC2025_valid/CCAC2025_valid/valid_zh_task2"
    valid_en_task1_src = f"{DATASET_ROOT}/CCAC2025_valid/CCAC2025_valid/valid_en_task1"
    valid_en_task2_src = f"{DATASET_ROOT}/CCAC2025_valid/CCAC2025_valid/valid_en_task2"
    
    valid_zh_task1_dst = f"{OUTPUT_DIR}/valid_zh_task1"
    valid_zh_task2_dst = f"{OUTPUT_DIR}/valid_zh_task2"
    valid_en_task1_dst = f"{OUTPUT_DIR}/valid_en_task1"
    valid_en_task2_dst = f"{OUTPUT_DIR}/valid_en_task2"
    
    count_v_zh_t1 = copy_images(valid_zh_task1_src, valid_zh_task1_dst)
    count_v_zh_t2 = copy_images(valid_zh_task2_src, valid_zh_task2_dst)
    count_v_en_t1 = copy_images(valid_en_task1_src, valid_en_task1_dst)
    count_v_en_t2 = copy_images(valid_en_task2_src, valid_en_task2_dst)
    
    print(f"  中文任务1: {count_v_zh_t1} 张 -> {valid_zh_task1_dst}")
    print(f"  中文任务2: {count_v_zh_t2} 张 -> {valid_zh_task2_dst}")
    print(f"  英文任务1: {count_v_en_t1} 张 -> {valid_en_task1_dst}")
    print(f"  英文任务2: {count_v_en_t2} 张 -> {valid_en_task2_dst}")
    
    # 测试集图片
    print("\n【测试集】")
    test_zh_task1_src = f"{DATASET_ROOT}/CCAC2025manga_test/CCAC2025test/test_zh_task1"
    test_zh_task2_src = f"{DATASET_ROOT}/CCAC2025manga_test/CCAC2025test/test_zh_task2"
    test_en_task1_src = f"{DATASET_ROOT}/CCAC2025manga_test/CCAC2025test/test_en_task1"
    test_en_task2_src = f"{DATASET_ROOT}/CCAC2025manga_test/CCAC2025test/test_en_task2"
    
    test_zh_task1_dst = f"{OUTPUT_DIR}/test_zh_task1"
    test_zh_task2_dst = f"{OUTPUT_DIR}/test_zh_task2"
    test_en_task1_dst = f"{OUTPUT_DIR}/test_en_task1"
    test_en_task2_dst = f"{OUTPUT_DIR}/test_en_task2"
    
    count_t_zh_t1 = copy_images(test_zh_task1_src, test_zh_task1_dst)
    count_t_zh_t2 = copy_images(test_zh_task2_src, test_zh_task2_dst)
    count_t_en_t1 = copy_images(test_en_task1_src, test_en_task1_dst)
    count_t_en_t2 = copy_images(test_en_task2_src, test_en_task2_dst)
    
    print(f"  中文任务1: {count_t_zh_t1} 张 -> {test_zh_task1_dst}")
    print(f"  中文任务2: {count_t_zh_t2} 张 -> {test_zh_task2_dst}")
    print(f"  英文任务1: {count_t_en_t1} 张 -> {test_en_task1_dst}")
    print(f"  英文任务2: {count_t_en_t2} 张 -> {test_en_task2_dst}")

def parse_desp(desp_str):
    """解析desp字段，处理Python列表字符串"""
    try:
        # 尝试用ast.literal_eval解析
        return ast.literal_eval(desp_str)
    except:
        # 如果失败，返回空列表
        return []

def build_task1_from_train():
    """从训练集构建任务1数据"""
    print("\n" + "=" * 70)
    print("【训练集】任务1：四格漫画逻辑理解")
    print("=" * 70)
    
    task1_data = []
    task1_zh = []
    task1_en = []
    
    # 处理中文训练集
    zh_csv = f"{DATASET_ROOT}/CCAC2025manga_train/CCAC2025manga_train/train_zh.csv"
    with open(zh_csv, 'r', encoding='gbk', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"  中文训练样本: {len(rows)} 条漫画")
    
    for row in rows:
        comic_id = row['漫画标签']
        shuffle_variants = [
            ('sp01', row.get('sp01', ''), row.get('sp01阅读顺序', '')),
            ('sp12', row.get('sp12', ''), row.get('sp12阅读顺序', '')),
            ('sp23', row.get('sp23', ''), row.get('sp23阅读顺序', '')),
            ('sprandom', row.get('sprandom', ''), row.get('sprandom阅读顺序', '')),
        ]
        
        for variant_name, img_file, correct_order in shuffle_variants:
            if not img_file or not correct_order:
                continue
            
            img_path = f"{OUTPUT_DIR}/train_zh/{img_file}"
            if not os.path.exists(img_path):
                continue
            
            rel_img_path = f"ccac2025_complete/train_zh/{img_file}"
            
            item = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>这是一组顺序被打乱的四格漫画。请分析漫画内容的逻辑顺序，输出正确的阅读顺序（用空格分隔的数字，如\"0 1 2 3\"表示先读第0格，然后第1格，然后第2格，最后第3格）。只输出数字顺序，不要其他解释。"
                    },
                    {
                        "from": "gpt",
                        "value": correct_order.strip()
                    }
                ],
                "images": [rel_img_path]
            }
            task1_data.append(item)
            task1_zh.append(item)
    
    print(f"  中文任务1样本: {len(task1_zh)} 条")
    
    # 处理英文训练集
    en_csv = f"{DATASET_ROOT}/CCAC2025manga_train/CCAC2025manga_train/train_en.csv"
    with open(en_csv, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"  英文训练样本: {len(rows)} 条漫画")
    
    for row in rows:
        comic_id = row['漫画标签']
        shuffle_variants = [
            ('sp01', row.get('sp01', ''), row.get('sp01阅读顺序', '')),
            ('sp12', row.get('sp12', ''), row.get('sp12阅读顺序', '')),
            ('sp23', row.get('sp23', ''), row.get('sp23阅读顺序', '')),
        ]
        
        for variant_name, img_path_in_csv, correct_order in shuffle_variants:
            if not img_path_in_csv or not correct_order:
                continue
            
            img_filename = os.path.basename(img_path_in_csv)
            img_path = f"{OUTPUT_DIR}/train_en/{img_filename}"
            
            if not os.path.exists(img_path):
                continue
            
            rel_img_path = f"ccac2025_complete/train_en/{img_filename}"
            
            item = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>This is a 4-panel comic with shuffled order. Please analyze the logical sequence of the comic panels and output the correct reading order (use space-separated numbers, e.g., \"0 1 2 3\" means read panel 0 first, then panel 1, then panel 2, and finally panel 3). Only output the number sequence without any explanation."
                    },
                    {
                        "from": "gpt",
                        "value": correct_order.strip()
                    }
                ],
                "images": [rel_img_path]
            }
            task1_data.append(item)
            task1_en.append(item)
    
    print(f"  英文任务1样本: {len(task1_en)} 条")
    print(f"  任务1训练总样本: {len(task1_data)} 条")
    
    return task1_data, task1_zh, task1_en

def build_task2_from_train():
    """从训练集构建任务2数据"""
    print("\n" + "=" * 70)
    print("【训练集】任务2：四格漫画上下文推理")
    print("=" * 70)
    
    task2_data = []
    task2_zh = []
    task2_en = []
    
    # 中文训练集
    zh_csv = f"{DATASET_ROOT}/CCAC2025manga_train/CCAC2025manga_train/train_zh.csv"
    with open(zh_csv, 'r', encoding='gbk', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"  中文训练样本: {len(rows)} 条漫画")
    
    for row in rows:
        comic_id = row['漫画标签']
        panel_descs = [
            row.get('panel0描述', ''),
            row.get('panel1描述', ''),
            row.get('panel2描述', ''),
            row.get('panel3描述', '')
        ]
        
        original_img = row.get('原始漫画', '')
        if not original_img:
            continue
        
        img_path = f"{OUTPUT_DIR}/train_zh/{original_img}"
        if not os.path.exists(img_path):
            continue
        
        rel_img_path = f"ccac2025_complete/train_zh/{original_img}"
        
        for mask_idx in range(4):
            if not panel_descs[mask_idx]:
                continue
            
            other_panels = [(i, desc) for i, desc in enumerate(panel_descs) if i != mask_idx and desc]
            if len(other_panels) < 3:
                continue
            
            context_desc = "\n".join([f"第{i}格描述：{desc}" for i, desc in other_panels])
            target_desc = panel_descs[mask_idx]
            
            item = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>这是一组四格漫画，其中第{mask_idx}格的内容被遮罩。根据漫画图片和其他3格的描述，请生成第{mask_idx}格的文本描述。\n\n{context_desc}\n\n请输出生成的第{mask_idx}格描述："
                    },
                    {
                        "from": "gpt",
                        "value": target_desc
                    }
                ],
                "images": [rel_img_path]
            }
            task2_data.append(item)
            task2_zh.append(item)
    
    print(f"  中文任务2样本: {len(task2_zh)} 条")
    
    # 英文训练集
    en_csv = f"{DATASET_ROOT}/CCAC2025manga_train/CCAC2025manga_train/train_en.csv"
    with open(en_csv, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"  英文训练样本: {len(rows)} 条漫画")
    
    for row in rows:
        comic_id = row['漫画标签']
        panel_descs = [
            row.get('panel0描述', ''),
            row.get('panel1描述', ''),
            row.get('panel2描述', ''),
            row.get('panel3描述', '')
        ]
        
        original_img_path = row.get('原始漫画', '')
        if not original_img_path:
            continue
        
        img_filename = os.path.basename(original_img_path)
        img_path = f"{OUTPUT_DIR}/train_en/{img_filename}"
        
        if not os.path.exists(img_path):
            continue
        
        rel_img_path = f"ccac2025_complete/train_en/{img_filename}"
        
        for mask_idx in range(4):
            if not panel_descs[mask_idx]:
                continue
            
            other_panels = [(i, desc) for i, desc in enumerate(panel_descs) if i != mask_idx and desc]
            if len(other_panels) < 3:
                continue
            
            context_desc = "\n".join([f"Panel {i} description: {desc}" for i, desc in other_panels])
            target_desc = panel_descs[mask_idx]
            
            item = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>This is a 4-panel comic where panel {mask_idx} is masked. Based on the comic image and descriptions of the other 3 panels, please generate a description for panel {mask_idx}.\n\n{context_desc}\n\nPlease generate the description for panel {mask_idx}:"
                    },
                    {
                        "from": "gpt",
                        "value": target_desc
                    }
                ],
                "images": [rel_img_path]
            }
            task2_data.append(item)
            task2_en.append(item)
    
    print(f"  英文任务2样本: {len(task2_en)} 条")
    print(f"  任务2训练总样本: {len(task2_data)} 条")
    
    return task2_data, task2_zh, task2_en

def build_valid_task1():
    """构建验证集任务1数据"""
    print("\n" + "=" * 70)
    print("【验证集】任务1：四格漫画逻辑理解")
    print("=" * 70)
    
    valid_data = []
    valid_zh = []
    valid_en = []
    
    # 中文验证集
    csv_path = f"{DATASET_ROOT}/CCAC2025_valid/CCAC2025_valid/valid_zh_task1.csv"
    with open(csv_path, 'r', encoding='gbk', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"  中文验证样本: {len(rows)} 条")
    
    for row in rows:
        comic_id = row['id']
        label = row.get('label', '')
        
        # 查找对应的图片
        img_file = None
        valid_img_dir = f"{OUTPUT_DIR}/valid_zh_task1"
        for ext in ['.jpg', '.jpeg', '.png']:
            test_file = f"{comic_id}{ext}"
            if os.path.exists(f"{valid_img_dir}/{test_file}"):
                img_file = test_file
                break
        
        if not img_file:
            continue
        
        rel_img_path = f"ccac2025_complete/valid_zh_task1/{img_file}"
        
        item = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>这是一组顺序被打乱的四格漫画。请分析漫画内容的逻辑顺序，输出正确的阅读顺序（用空格分隔的数字，如\"0 1 2 3\"表示先读第0格，然后第1格，然后第2格，最后第3格）。只输出数字顺序，不要其他解释。"
                },
                {
                    "from": "gpt",
                    "value": label.strip() if label else ""
                }
            ],
            "images": [rel_img_path],
            "metadata": {
                "comic_id": comic_id,
                "split": "valid",
                "language": "zh",
                "task": "task1"
            }
        }
        valid_data.append(item)
        valid_zh.append(item)
    
    print(f"  中文验证样本(有效): {len(valid_zh)} 条")
    
    # 英文验证集
    csv_path = f"{DATASET_ROOT}/CCAC2025_valid/CCAC2025_valid/valid_en_task1.csv"
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"  英文验证样本: {len(rows)} 条")
    
    for row in rows:
        comic_id = row['id']
        label = row.get('label', '')
        
        img_file = None
        valid_img_dir = f"{OUTPUT_DIR}/valid_en_task1"
        for ext in ['.jpg', '.jpeg', '.png']:
            test_file = f"{comic_id}{ext}"
            if os.path.exists(f"{valid_img_dir}/{test_file}"):
                img_file = test_file
                break
        
        if not img_file:
            continue
        
        rel_img_path = f"ccac2025_complete/valid_en_task1/{img_file}"
        
        item = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>This is a 4-panel comic with shuffled order. Please analyze the logical sequence of the comic panels and output the correct reading order (use space-separated numbers, e.g., \"0 1 2 3\" means read panel 0 first, then panel 1, then panel 2, and finally panel 3). Only output the number sequence without any explanation."
                },
                {
                    "from": "gpt",
                    "value": label.strip() if label else ""
                }
            ],
            "images": [rel_img_path],
            "metadata": {
                "comic_id": comic_id,
                "split": "valid",
                "language": "en",
                "task": "task1"
            }
        }
        valid_data.append(item)
        valid_en.append(item)
    
    print(f"  英文验证样本(有效): {len(valid_en)} 条")
    print(f"  验证集任务1总样本: {len(valid_data)} 条")
    
    return valid_data, valid_zh, valid_en

def build_valid_task2():
    """构建验证集任务2数据"""
    print("\n" + "=" * 70)
    print("【验证集】任务2：四格漫画上下文推理")
    print("=" * 70)
    
    valid_data = []
    valid_zh = []
    valid_en = []
    
    # 中文验证集
    csv_path = f"{DATASET_ROOT}/CCAC2025_valid/CCAC2025_valid/valid_zh_task2.csv"
    with open(csv_path, 'r', encoding='gbk', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"  中文验证样本: {len(rows)} 条")
    
    for row in rows:
        comic_id = row['id']
        desp_str = row.get('desp', '')
        mask_panel = row.get('mask_panel', '')
        label = row.get('label', '')
        
        # 解析desp字段
        panel_descs = parse_desp(desp_str)
        if not panel_descs or len(panel_descs) != 4:
            continue
        
        # 查找图片
        img_file = None
        valid_img_dir = f"{OUTPUT_DIR}/valid_zh_task2"
        for ext in ['.jpg', '.jpeg', '.png']:
            test_file = f"{comic_id}{ext}"
            if os.path.exists(f"{valid_img_dir}/{test_file}"):
                img_file = test_file
                break
        
        if not img_file:
            continue
        
        rel_img_path = f"ccac2025_complete/valid_zh_task2/{img_file}"
        
        # 构建上下文描述
        mask_idx = int(mask_panel) if mask_panel.isdigit() else 0
        other_panels = [(i, desc) for i, desc in enumerate(panel_descs) if i != mask_idx and desc and desc != 'masked']
        
        if len(other_panels) < 3:
            continue
        
        context_desc = "\n".join([f"第{i}格描述：{desc}" for i, desc in other_panels])
        
        item = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>这是一组四格漫画，其中第{mask_idx}格的内容被遮罩。根据漫画图片和其他3格的描述，请生成第{mask_idx}格的文本描述。\n\n{context_desc}\n\n请输出生成的第{mask_idx}格描述："
                },
                {
                    "from": "gpt",
                    "value": label
                }
            ],
            "images": [rel_img_path],
            "metadata": {
                "comic_id": comic_id,
                "mask_panel": mask_idx,
                "split": "valid",
                "language": "zh",
                "task": "task2"
            }
        }
        valid_data.append(item)
        valid_zh.append(item)
    
    print(f"  中文验证样本(有效): {len(valid_zh)} 条")
    
    # 英文验证集
    csv_path = f"{DATASET_ROOT}/CCAC2025_valid/CCAC2025_valid/valid_en_task2.csv"
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"  英文验证样本: {len(rows)} 条")
    
    for row in rows:
        comic_id = row['id']
        desp_str = row.get('desp', '')
        mask_panel = row.get('mask_panel', '')
        label = row.get('label', '')
        
        panel_descs = parse_desp(desp_str)
        if not panel_descs or len(panel_descs) != 4:
            continue
        
        img_file = None
        valid_img_dir = f"{OUTPUT_DIR}/valid_en_task2"
        for ext in ['.jpg', '.jpeg', '.png']:
            test_file = f"{comic_id}{ext}"
            if os.path.exists(f"{valid_img_dir}/{test_file}"):
                img_file = test_file
                break
        
        if not img_file:
            continue
        
        rel_img_path = f"ccac2025_complete/valid_en_task2/{img_file}"
        
        mask_idx = int(mask_panel) if mask_panel.isdigit() else 0
        other_panels = [(i, desc) for i, desc in enumerate(panel_descs) if i != mask_idx and desc and desc != 'masked']
        
        if len(other_panels) < 3:
            continue
        
        context_desc = "\n".join([f"Panel {i} description: {desc}" for i, desc in other_panels])
        
        item = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>This is a 4-panel comic where panel {mask_idx} is masked. Based on the comic image and descriptions of the other 3 panels, please generate a description for panel {mask_idx}.\n\n{context_desc}\n\nPlease generate the description for panel {mask_idx}:"
                },
                {
                    "from": "gpt",
                    "value": label
                }
            ],
            "images": [rel_img_path],
            "metadata": {
                "comic_id": comic_id,
                "mask_panel": mask_idx,
                "split": "valid",
                "language": "en",
                "task": "task2"
            }
        }
        valid_data.append(item)
        valid_en.append(item)
    
    print(f"  英文验证样本(有效): {len(valid_en)} 条")
    print(f"  验证集任务2总样本: {len(valid_data)} 条")
    
    return valid_data, valid_zh, valid_en

def build_test_task1():
    """构建测试集任务1数据（无标签，用于提交）"""
    print("\n" + "=" * 70)
    print("【测试集】任务1：四格漫画逻辑理解（无标签）")
    print("=" * 70)
    
    test_data = []
    test_zh = []
    test_en = []
    
    # 中文测试集
    csv_path = f"{DATASET_ROOT}/CCAC2025manga_test/CCAC2025test/test_zh_task1.csv"
    with open(csv_path, 'r', encoding='gbk', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"  中文测试样本: {len(rows)} 条")
    
    for row in rows:
        comic_id = row['id']
        random_flag = row.get('random', 'FALSE')
        
        img_file = None
        test_img_dir = f"{OUTPUT_DIR}/test_zh_task1"
        for ext in ['.jpg', '.jpeg', '.png']:
            test_file = f"{comic_id}{ext}"
            if os.path.exists(f"{test_img_dir}/{test_file}"):
                img_file = test_file
                break
        
        if not img_file:
            continue
        
        rel_img_path = f"ccac2025_complete/test_zh_task1/{img_file}"
        
        item = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>这是一组顺序被打乱的四格漫画。请分析漫画内容的逻辑顺序，输出正确的阅读顺序（用空格分隔的数字，如\"0 1 2 3\"表示先读第0格，然后第1格，然后第2格，最后第3格）。只输出数字顺序，不要其他解释。"
                },
                {
                    "from": "gpt",
                    "value": ""  # 测试集无标签
                }
            ],
            "images": [rel_img_path],
            "metadata": {
                "comic_id": comic_id,
                "split": "test",
                "language": "zh",
                "task": "task1",
                "random": random_flag
            }
        }
        test_data.append(item)
        test_zh.append(item)
    
    print(f"  中文测试样本(有效): {len(test_zh)} 条")
    
    # 英文测试集
    csv_path = f"{DATASET_ROOT}/CCAC2025manga_test/CCAC2025test/test_en_task1.csv"
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"  英文测试样本: {len(rows)} 条")
    
    for row in rows:
        comic_id = row['id']
        random_flag = row.get('random', 'FALSE')
        
        img_file = None
        test_img_dir = f"{OUTPUT_DIR}/test_en_task1"
        for ext in ['.jpg', '.jpeg', '.png']:
            test_file = f"{comic_id}{ext}"
            if os.path.exists(f"{test_img_dir}/{test_file}"):
                img_file = test_file
                break
        
        if not img_file:
            continue
        
        rel_img_path = f"ccac2025_complete/test_en_task1/{img_file}"
        
        item = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>This is a 4-panel comic with shuffled order. Please analyze the logical sequence of the comic panels and output the correct reading order (use space-separated numbers, e.g., \"0 1 2 3\" means read panel 0 first, then panel 1, then panel 2, and finally panel 3). Only output the number sequence without any explanation."
                },
                {
                    "from": "gpt",
                    "value": ""  # 测试集无标签
                }
            ],
            "images": [rel_img_path],
            "metadata": {
                "comic_id": comic_id,
                "split": "test",
                "language": "en",
                "task": "task1",
                "random": random_flag
            }
        }
        test_data.append(item)
        test_en.append(item)
    
    print(f"  英文测试样本(有效): {len(test_en)} 条")
    print(f"  测试集任务1总样本: {len(test_data)} 条")
    
    return test_data, test_zh, test_en

def build_test_task2():
    """构建测试集任务2数据（无标签，用于提交）"""
    print("\n" + "=" * 70)
    print("【测试集】任务2：四格漫画上下文推理（无标签）")
    print("=" * 70)
    
    test_data = []
    test_zh = []
    test_en = []
    
    # 中文测试集
    csv_path = f"{DATASET_ROOT}/CCAC2025manga_test/CCAC2025test/test_zh_task2.csv"
    with open(csv_path, 'r', encoding='gbk', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"  中文测试样本: {len(rows)} 条")
    
    for row in rows:
        comic_id = row['id']
        desp_str = row.get('desp', '')
        mask_panel = row.get('mask_panel', '')
        
        panel_descs = parse_desp(desp_str)
        if not panel_descs or len(panel_descs) != 4:
            continue
        
        img_file = None
        test_img_dir = f"{OUTPUT_DIR}/test_zh_task2"
        for ext in ['.jpg', '.jpeg', '.png']:
            test_file = f"{comic_id}{ext}"
            if os.path.exists(f"{test_img_dir}/{test_file}"):
                img_file = test_file
                break
        
        if not img_file:
            continue
        
        rel_img_path = f"ccac2025_complete/test_zh_task2/{img_file}"
        
        mask_idx = int(mask_panel) if mask_panel.isdigit() else 0
        other_panels = [(i, desc) for i, desc in enumerate(panel_descs) if i != mask_idx and desc and desc != 'masked']
        
        if len(other_panels) < 3:
            continue
        
        context_desc = "\n".join([f"第{i}格描述：{desc}" for i, desc in other_panels])
        
        item = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>这是一组四格漫画，其中第{mask_idx}格的内容被遮罩。根据漫画图片和其他3格的描述，请生成第{mask_idx}格的文本描述。\n\n{context_desc}\n\n请输出生成的第{mask_idx}格描述："
                },
                {
                    "from": "gpt",
                    "value": ""  # 测试集无标签
                }
            ],
            "images": [rel_img_path],
            "metadata": {
                "comic_id": comic_id,
                "mask_panel": mask_idx,
                "split": "test",
                "language": "zh",
                "task": "task2"
            }
        }
        test_data.append(item)
        test_zh.append(item)
    
    print(f"  中文测试样本(有效): {len(test_zh)} 条")
    
    # 英文测试集
    csv_path = f"{DATASET_ROOT}/CCAC2025manga_test/CCAC2025test/test_en_task2.csv"
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"  英文测试样本: {len(rows)} 条")
    
    for row in rows:
        comic_id = row['id']
        desp_str = row.get('desp', '')
        mask_panel = row.get('mask_panel', '')
        
        panel_descs = parse_desp(desp_str)
        if not panel_descs or len(panel_descs) != 4:
            continue
        
        img_file = None
        test_img_dir = f"{OUTPUT_DIR}/test_en_task2"
        for ext in ['.jpg', '.jpeg', '.png']:
            test_file = f"{comic_id}{ext}"
            if os.path.exists(f"{test_img_dir}/{test_file}"):
                img_file = test_file
                break
        
        if not img_file:
            continue
        
        rel_img_path = f"ccac2025_complete/test_en_task2/{img_file}"
        
        mask_idx = int(mask_panel) if mask_panel.isdigit() else 0
        other_panels = [(i, desc) for i, desc in enumerate(panel_descs) if i != mask_idx and desc and desc != 'masked']
        
        if len(other_panels) < 3:
            continue
        
        context_desc = "\n".join([f"Panel {i} description: {desc}" for i, desc in other_panels])
        
        item = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>This is a 4-panel comic where panel {mask_idx} is masked. Based on the comic image and descriptions of the other 3 panels, please generate a description for panel {mask_idx}.\n\n{context_desc}\n\nPlease generate the description for panel {mask_idx}:"
                },
                {
                    "from": "gpt",
                    "value": ""  # 测试集无标签
                }
            ],
            "images": [rel_img_path],
            "metadata": {
                "comic_id": comic_id,
                "mask_panel": mask_idx,
                "split": "test",
                "language": "en",
                "task": "task2"
            }
        }
        test_data.append(item)
        test_en.append(item)
    
    print(f"  英文测试样本(有效): {len(test_en)} 条")
    print(f"  测试集任务2总样本: {len(test_data)} 条")
    
    return test_data, test_zh, test_en

def save_datasets(all_data):
    """保存所有数据集"""
    print("\n" + "=" * 70)
    print("保存数据集文件")
    print("=" * 70)
    
    datasets_to_save = [
        # 训练集
        ("train_task1.json", all_data['train_task1']),
        ("train_task1_zh.json", all_data['train_task1_zh']),
        ("train_task1_en.json", all_data['train_task1_en']),
        ("train_task2.json", all_data['train_task2']),
        ("train_task2_zh.json", all_data['train_task2_zh']),
        ("train_task2_en.json", all_data['train_task2_en']),
        # 验证集
        ("valid_task1.json", all_data['valid_task1']),
        ("valid_task1_zh.json", all_data['valid_task1_zh']),
        ("valid_task1_en.json", all_data['valid_task1_en']),
        ("valid_task2.json", all_data['valid_task2']),
        ("valid_task2_zh.json", all_data['valid_task2_zh']),
        ("valid_task2_en.json", all_data['valid_task2_en']),
        # 测试集
        ("test_task1.json", all_data['test_task1']),
        ("test_task1_zh.json", all_data['test_task1_zh']),
        ("test_task1_en.json", all_data['test_task1_en']),
        ("test_task2.json", all_data['test_task2']),
        ("test_task2_zh.json", all_data['test_task2_zh']),
        ("test_task2_en.json", all_data['test_task2_en']),
    ]
    
    for filename, data in datasets_to_save:
        if data:
            filepath = f"{OUTPUT_DIR}/{filename}"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  ✓ {filename}: {len(data)} 条样本")

def generate_dataset_info():
    """生成dataset_info.json配置"""
    configs = {}
    
    prefixes = [
        "train_task1", "train_task1_zh", "train_task1_en",
        "train_task2", "train_task2_zh", "train_task2_en",
        "valid_task1", "valid_task1_zh", "valid_task1_en",
        "valid_task2", "valid_task2_zh", "valid_task2_en",
        "test_task1", "test_task1_zh", "test_task1_en",
        "test_task2", "test_task2_zh", "test_task2_en",
    ]
    
    for prefix in prefixes:
        configs[f"ccac2025_{prefix}"] = {
            "file_name": f"{prefix}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images"
            }
        }
    
    return configs

def update_dataset_info(configs):
    """更新dataset_info.json"""
    existing_config_path = f"{LLAMA_FACTORY_DATA}/dataset_info.json"
    
    existing_config = {}
    if os.path.exists(existing_config_path):
        with open(existing_config_path, 'r', encoding='utf-8') as f:
            existing_config = json.load(f)
    
    existing_config.update(configs)
    
    with open(existing_config_path, 'w', encoding='utf-8') as f:
        json.dump(existing_config, f, ensure_ascii=False, indent=2)
    
    # 保存备份
    with open(f"{OUTPUT_DIR}/dataset_info.json", 'w', encoding='utf-8') as f:
        json.dump(configs, f, ensure_ascii=False, indent=2)
    
    print(f"\n  ✓ 更新配置: {existing_config_path}")

def print_summary(all_data):
    """打印统计信息"""
    print("\n" + "=" * 70)
    print("数据集构建完成！")
    print("=" * 70)
    
    print(f"\n📁 数据集位置: {OUTPUT_DIR}")
    
    print("\n📊 数据统计:")
    print("\n【训练集】")
    print(f"  任务1 - 总计: {len(all_data['train_task1'])} 条 (中文: {len(all_data['train_task1_zh'])}, 英文: {len(all_data['train_task1_en'])})")
    print(f"  任务2 - 总计: {len(all_data['train_task2'])} 条 (中文: {len(all_data['train_task2_zh'])}, 英文: {len(all_data['train_task2_en'])})")
    
    print("\n【验证集】")
    print(f"  任务1 - 总计: {len(all_data['valid_task1'])} 条 (中文: {len(all_data['valid_task1_zh'])}, 英文: {len(all_data['valid_task1_en'])})")
    print(f"  任务2 - 总计: {len(all_data['valid_task2'])} 条 (中文: {len(all_data['valid_task2_zh'])}, 英文: {len(all_data['valid_task2_en'])})")
    
    print("\n【测试集】")
    print(f"  任务1 - 总计: {len(all_data['test_task1'])} 条 (中文: {len(all_data['test_task1_zh'])}, 英文: {len(all_data['test_task1_en'])})")
    print(f"  任务2 - 总计: {len(all_data['test_task2'])} 条 (中文: {len(all_data['test_task2_zh'])}, 英文: {len(all_data['test_task2_en'])})")
    
    print("\n📋 可用的数据集名称 (共18个):")
    print("\n  训练集:")
    print("    ccac2025_train_task1, ccac2025_train_task1_zh, ccac2025_train_task1_en")
    print("    ccac2025_train_task2, ccac2025_train_task2_zh, ccac2025_train_task2_en")
    print("\n  验证集:")
    print("    ccac2025_valid_task1, ccac2025_valid_task1_zh, ccac2025_valid_task1_en")
    print("    ccac2025_valid_task2, ccac2025_valid_task2_zh, ccac2025_valid_task2_en")
    print("\n  测试集:")
    print("    ccac2025_test_task1, ccac2025_test_task1_zh, ccac2025_test_task1_en")
    print("    ccac2025_test_task2, ccac2025_test_task2_zh, ccac2025_test_task2_en")
    
    print("\n💡 使用示例:")
    print("  训练: dataset=ccac2025_train_task1")
    print("  验证: dataset=ccac2025_valid_task1")
    print("  测试: dataset=ccac2025_test_task1 (无标签，用于生成提交文件)")
    
    print("\n⚠️  注意:")
    print("  1. 测试集没有标签(gpt字段为空)，用于生成预测结果提交比赛")
    print("  2. 验证集有标签，可用于评估模型性能")
    print("  3. 整个数据集包含图片，约 500MB+")
    
    print("=" * 70)

def main():
    print("=" * 70)
    print("构建CCAC2025完整数据集 (训练集+验证集+测试集)")
    print("=" * 70)
    
    ensure_dir(OUTPUT_DIR)
    
    # 复制所有图片
    copy_all_images()
    
    # 构建训练集
    train_task1, train_task1_zh, train_task1_en = build_task1_from_train()
    train_task2, train_task2_zh, train_task2_en = build_task2_from_train()
    
    # 构建验证集
    valid_task1, valid_task1_zh, valid_task1_en = build_valid_task1()
    valid_task2, valid_task2_zh, valid_task2_en = build_valid_task2()
    
    # 构建测试集
    test_task1, test_task1_zh, test_task1_en = build_test_task1()
    test_task2, test_task2_zh, test_task2_en = build_test_task2()
    
    # 汇总数据
    all_data = {
        'train_task1': train_task1, 'train_task1_zh': train_task1_zh, 'train_task1_en': train_task1_en,
        'train_task2': train_task2, 'train_task2_zh': train_task2_zh, 'train_task2_en': train_task2_en,
        'valid_task1': valid_task1, 'valid_task1_zh': valid_task1_zh, 'valid_task1_en': valid_task1_en,
        'valid_task2': valid_task2, 'valid_task2_zh': valid_task2_zh, 'valid_task2_en': valid_task2_en,
        'test_task1': test_task1, 'test_task1_zh': test_task1_zh, 'test_task1_en': test_task1_en,
        'test_task2': test_task2, 'test_task2_zh': test_task2_zh, 'test_task2_en': test_task2_en,
    }
    
    # 保存数据集
    save_datasets(all_data)
    
    # 更新配置
    configs = generate_dataset_info()
    update_dataset_info(configs)
    
    # 打印统计
    print_summary(all_data)

if __name__ == "__main__":
    main()
