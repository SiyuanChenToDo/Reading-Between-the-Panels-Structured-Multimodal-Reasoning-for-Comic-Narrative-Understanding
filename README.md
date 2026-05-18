# Reading Between the Panels: Structured Multimodal Reasoning for Comic Narrative Understanding

<p align="center">
  <img src="https://img.shields.io/badge/Task-Panel%20Order%20%26%20Captioning-blue" alt="Task">
  <img src="https://img.shields.io/badge/Framework-PyTorch%20%7C%20LlamaFactory-ee4c2c" alt="Framework">
  <img src="https://img.shields.io/badge/Dataset-CCAC2025-green" alt="Dataset">
  <img src="https://img.shields.io/badge/License-Apache%202.0-yellow" alt="License">
</p>

<p align="center">
  <b>四格漫画结构化多模态推理</b> —— CCAC2025 四格漫画理解比赛核心方案
</p>

---

## 📖 Introduction

This repository contains the core implementation for **CCAC2025 (Comic Narrative Understanding Competition)**, which addresses two challenging tasks on four-panel comic understanding:

- **Task 1: Panel Order Prediction** — Given a shuffled four-panel comic, predict the correct reading sequence (e.g., `"0 2 1 3"`).
- **Task 2: Masked Panel Captioning** — Given a comic with one masked panel and descriptions of the other three, generate a text description for the missing panel.

Unlike simple image classification approaches, our solution explores **structured multimodal reasoning** through multi-scale encoding, contrastive learning, and story-aware attention mechanisms, built on top of the LlamaFactory framework for efficient vision-language model fine-tuning.

---

## 🏗️ Project Structure

```
.
├── comic_innovation/                    # Core innovation modules
│   ├── balance_task1_data.py            # Permutation-level data balancing
│   ├── contrastive_loss.py              # Panel-wise contrastive learning (hard negative mining)
│   ├── generate_missing_perms.py        # Synthetic sample generation for missing permutations
│   ├── panel_extractor.py               # 2×2 composite image splitting utility
│   ├── preprocess_multiscale.py         # Multi-scale dataset construction (composite + 4 panels)
│   └── story_attention.py               # Story-aware attention with Sinkhorn permutation learning
│
├── panel_order_mm_pipeline/             # Baseline reproduction & ablation studies
│   ├── data/                            # Preprocessed metadata & vocab
│   ├── data_e2e/                        # End-to-end dataset (panel crops)
│   ├── data_split/                      # Panel-split features & images
│   ├── src/
│   │   ├── prepare_data.py              # Panel splitting + ViT feature extraction
│   │   ├── train.py / train_v2~v4.py    # Multimodal temporal ordering models
│   │   └── evaluate.py / evaluate_v2~v4.py
│   ├── scripts/run_all.sh               # One-command baseline pipeline
│   ├── requirements.txt
│   └── README.md                        # Detailed baseline documentation
│
├── panel_order_vit_pipeline/            # ViT-only baseline (label mapping)
│
├── LlamaFactory/                        # Fine-tuning framework (custom configs + dataset)
│   ├── data/ccac2025_complete/          # Full CCAC2025 dataset (ShareGPT format)
│   │   ├── task1/{train,valid,test}/    # Task 1 images & JSON annotations
│   │   ├── task2/{train,valid,test}/    # Task 2 images & JSON annotations
│   │   ├── joint/                       # Joint training datasets
│   │   └── dataset_info.json            # Dataset registration for LlamaFactory
│   ├── train_*.yaml                     # Task-specific training configurations
│   ├── src/                             # LlamaFactory source (minimal, ~3MB)
│   └── run_task2_push_train.sh          # Example training script
│
└── README.md                            # This file
```

---

## 🎯 Tasks

### Task 1: Panel Order Prediction

**Input:** A shuffled 2×2 four-panel comic image.  
**Output:** The correct reading order as a space-separated sequence (e.g., `"0 1 3 2"`).  
**Metric:** Macro-F1 (average of 4 position-level accuracies).

**Key Challenge:** The dataset exhibits severe language imbalance — English samples (~80%) contain only 3 dominant permutations, while Chinese samples (~20%) span all 22 possible permutations, making it extremely difficult to generalize across languages.

### Task 2: Masked Panel Captioning

**Input:** A four-panel comic with one panel masked + textual descriptions of the remaining 3 panels.  
**Output:** A fluent textual description of the masked panel.  
**Metric:** ROUGE-L, BLEU.

---

## 🧠 Core Innovations

### 1. Multi-Scale Encoding (`preprocess_multiscale.py`)
Constructs multi-modal inputs by combining the **full composite image** with **4 individual panel crops**, enabling the model to perceive both global layout and local details.

### 2. Panel-Wise Contrastive Learning (`contrastive_loss.py`)
- **Hard negative mining**: Retains only the most challenging negative samples.
- **FIFO ring buffer queue**: Efficient negative sample storage with learnable temperature (clamped ≥ 0.03).
- **Narrative-aware positives**: Extends positive pairs beyond adjacent panels to narrative-distance ≤ 2 (e.g., beginning↔transition, development↔conclusion).

### 3. Story-Aware Attention with Sinkhorn (`story_attention.py`)
A lightweight (~10M params) permutation learning module that replaces coarse 24-way classification:
- **Sinkhorn operator**: Differentiable doubly-stochastic matrix learning.
- **Vectorized target construction**: Efficient label generation for all 24 permutations.
- **Auxiliary warmup**: Gradual introduction of auxiliary losses to stabilize early training.

### 4. Data Balancing & Augmentation
- **`balance_task1_data.py`**: Permutation-level oversampling to combat distribution bias.
- **`generate_missing_perms.py`**: Synthesizes composite images for under-represented permutations by recombining existing panels, achieving full 24-permutation coverage without polluting the validation set.

---

## 📊 Dataset

We provide the **complete CCAC2025 dataset** in LlamaFactory-compatible [ShareGPT multimodal format](https://github.com/hiyouga/LLaMA-Factory):

| Split | Task 1 | Task 2 | Language | Labeled |
|-------|--------|--------|----------|---------|
| Train | 1,923 | 2,436 | zh + en | ✅ |
| Valid | 203 | 198 | zh + en | ✅ |
| Test | 191 | 201 | zh + en | ❌ (submission only) |

**Dataset location:** `LlamaFactory/data/ccac2025_complete/`  
**Registration file:** `LlamaFactory/data/ccac2025_complete/dataset_info.json`

### Data Format Example (Task 1)

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<image>这是一组顺序被打乱的四格漫画。请分析漫画内容的逻辑顺序，输出正确的阅读顺序（用空格分隔的数字，如\"0 1 2 3\"）。只输出数字顺序，不要其他解释。"
    },
    {
      "from": "gpt",
      "value": "0 2 3 1"
    }
  ],
  "images": ["ccac2025_complete/task1/valid/zh/216.jpg"],
  "metadata": {"comic_id": "216", "language": "zh", "task": "task1"}
}
```

---

## 🚀 Quick Start

### Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/SiyuanChenToDo/Reading-Between-the-Panels-Structured-Multimodal-Reasoning-for-Comic-Narrative-Understanding.git
cd Reading-Between-the-Panels-Structured-Multimodal-Reasoning-for-Comic-Narrative-Understanding

# 2. Install baseline dependencies
pip install -r panel_order_mm_pipeline/requirements.txt

# 3. Install LlamaFactory (for VLM fine-tuning)
cd LlamaFactory
pip install -e .
cd ..
```

### Option A: Run Baseline Pipeline

The `panel_order_mm_pipeline/` provides a complete standalone baseline using ViT + LSTM + Transformer + ListMLE:

```bash
cd panel_order_mm_pipeline
bash scripts/run_all.sh
```

Or step-by-step:

```bash
# Step 1: Split composites & extract ViT features
python src/prepare_data.py

# Step 2: Train multimodal temporal model (~1-2 min on GPU)
python src/train.py

# Step 3: Evaluate
python src/evaluate.py
```

### Option B: Fine-tune Qwen2.5-VL with LlamaFactory

We provide a family of training configurations under `LlamaFactory/train_*.yaml`:

| Config | Description |
|--------|-------------|
| `train_ccac2025_task1_sota.yaml` | Task 1 SOTA config (LoRA rank=32, full modules) |
| `train_task1_ms_v3_full.yaml` | Full innovation stack (contrastive + story + balanced data) |
| `train_task1_ms_v3_baseline.yaml` | Baseline ablation |
| `train_task2_simple.yaml` | Task 2 simple fine-tuning |
| `train_task2_style_aligned.yaml` | Task 2 with style-aligned prompting |

**Example — Task 1 with best config:**

```bash
cd LlamaFactory

# Download Qwen2.5-VL-7B-Instruct first (not included in this repo)
# HuggingFace: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

# Update model path in yaml to your local path, then:
llamafactory-cli train train_ccac2025_task1_sota.yaml
```

**Example — Task 2:**

```bash
llamafactory-cli train train_task2_style_aligned.yaml
```

### Using Innovation Modules

Integrate `comic_innovation/` modules into your training loop:

```python
from comic_innovation.story_attention import StoryAwareModule
from comic_innovation.contrastive_loss import PanelContrastiveLoss

# Attach to your vision-language model
story_module = StoryAwareModule(hidden_dim=3584, num_panels=4)
cl_loss = PanelContrastiveLoss(hidden_dim=3584, proj_dim=128)
```

---

## 📈 Results

### Task 1: Panel Order Prediction (Baseline Pipeline)

| Dataset | Language | Samples | Macro-F1 | Exact Match | Pos0 | Pos1 | Pos2 | Pos3 |
|---------|----------|---------|----------|-------------|------|------|------|------|
| **Valid** | Chinese | 75 | **31.33%** | 0.00% | 21.33% | 57.33% | 34.67% | 12.00% |
| **Valid** | English | 128 | **51.56%** | 3.12% | 68.75% | 39.84% | 34.38% | 63.28% |
| **Valid** | Overall | 203 | **44.09%** | 1.97% | 51.23% | 46.31% | 34.48% | 44.33% |

> **Observation:** The model suffers from severe language imbalance. English performance is significantly better due to limited permutation diversity (only 3 dominant classes), while Chinese generalization remains challenging with 22 permutations.

### Task 1: Qwen2.5-VL LoRA Fine-tuning (LlamaFactory)

With multi-scale inputs + contrastive learning + story-aware attention, our best LoRA fine-tuned model achieves strong improvements over the frozen-feature baseline. See `LlamaFactory/train_task1_ms_v3_full.yaml` for the full configuration.

---

## 📝 Notes

- **Pre-trained model weights** (e.g., Qwen2.5-VL-7B) are **not included** in this repository. Please download them from HuggingFace / ModelScope separately.
- **Trained checkpoints** (`*.pt`, `*.safetensors`) are excluded to keep the repository size manageable. Training scripts and configurations are provided for full reproducibility.
- The `panel_order_mm_pipeline/outputs/` directory contains evaluation JSONs and training logs; model weights can be regenerated by running the training scripts.

---

## 📚 Citation

If you find this work useful, please consider citing the CCAC2025 dataset and related papers:

```bibtex
@inproceedings{ccac2025,
  title={CCAC2025: Comic Narrative Understanding Challenge},
  year={2025},
  organization={CCAC}
}
```

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).  
The LlamaFactory framework is also under [Apache License 2.0](https://github.com/hiyouga/LLaMA-Factory/blob/main/LICENSE).

---

<p align="center">
  <i>"Reading between the panels — where visual storytelling meets structured reasoning."</i>
</p>
