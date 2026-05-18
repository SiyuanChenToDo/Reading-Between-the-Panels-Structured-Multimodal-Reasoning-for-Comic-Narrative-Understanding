# CCAC2025 完整数据集 (LlamaFactory适配版)

## 数据集简介

本数据集是为CCAC2025四格漫画理解比赛构建的完整多模态数据集，包含**训练集、验证集、测试集**，适配LlamaFactory框架进行微调和评估。

### 任务说明

**任务1：四格漫画逻辑理解**
- 输入：顺序被打乱的四格漫画图片
- 输出：正确的阅读顺序（如 `"0 1 2 3"`）
- 评估指标：Macro-F1 score

**任务2：四格漫画上下文推理**
- 输入：被mask一个panel的漫画 + 其他3个panel的描述
- 输出：被mask panel的文本描述
- 评估指标：ROUGE-L, BLEU

---

## 数据集结构

```
ccac2025_complete/
├── train_task1.json              # 任务1训练集（1923条）
├── train_task1_zh.json           # 任务1训练集-中文（384条）
├── train_task1_en.json           # 任务1训练集-英文（1539条）
├── train_task2.json              # 任务2训练集（2436条）
├── train_task2_zh.json           # 任务2训练集-中文（384条）
├── train_task2_en.json           # 任务2训练集-英文（2052条）
├── valid_task1.json              # 任务1验证集（203条）✅有标签
├── valid_task1_zh.json           # 任务1验证集-中文（75条）
├── valid_task1_en.json           # 任务1验证集-英文（128条）
├── valid_task2.json              # 任务2验证集（198条）✅有标签
├── valid_task2_zh.json           # 任务2验证集-中文（73条）
├── valid_task2_en.json           # 任务2验证集-英文（125条）
├── test_task1.json               # 任务1测试集（191条）❌无标签
├── test_task1_zh.json            # 任务1测试集-中文（63条）
├── test_task1_en.json            # 任务1测试集-英文（128条）
├── test_task2.json               # 任务2测试集（201条）❌无标签
├── test_task2_zh.json            # 任务2测试集-中文（74条）
├── test_task2_en.json            # 任务2测试集-英文（127条）
├── dataset_info.json             # LlamaFactory配置文件
├── train_zh/                     # 训练集中文图片（480张）
├── train_en/                     # 训练集英文图片（2052张）
├── valid_zh_task1/               # 验证集中文任务1图片（75张）
├── valid_zh_task2/               # 验证集中文任务2图片（73张）
├── valid_en_task1/               # 验证集英文任务1图片（128张）
├── valid_en_task2/               # 验证集英文任务2图片（125张）
├── test_zh_task1/                # 测试集中文任务1图片（63张）
├── test_zh_task2/                # 测试集中文任务2图片（75张）
├── test_en_task1/                # 测试集英文任务1图片（128张）
└── test_en_task2/                # 测试集英文任务2图片（127张）
```

---

## 数据统计

| 数据集 | 任务1 | 任务2 | 说明 |
|--------|-------|-------|------|
| **训练集** | 1923条 | 2436条 | 用于训练模型 |
| - 中文 | 384条 | 384条 | |
| - 英文 | 1539条 | 2052条 | |
| **验证集** | 203条 | 198条 | ✅ 有标签，用于评估 |
| - 中文 | 75条 | 73条 | |
| - 英文 | 128条 | 125条 | |
| **测试集** | 191条 | 201条 | ❌ 无标签，用于提交 |
| - 中文 | 63条 | 74条 | |
| - 英文 | 128条 | 127条 | |

---

## 数据格式

使用LlamaFactory支持的 **ShareGPT多模态格式**：

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<image>用户提示词"
    },
    {
      "from": "gpt",
      "value": "模型回答"
    }
  ],
  "images": ["图片相对路径"],
  "metadata": {
    "comic_id": "漫画ID",
    "split": "train/valid/test",
    "language": "zh/en",
    "task": "task1/task2"
  }
}
```

### 任务1示例

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<image>这是一组顺序被打乱的四格漫画。请分析漫画内容的逻辑顺序，输出正确的阅读顺序（用空格分隔的数字，如\"0 1 2 3\"表示先读第0格，然后第1格，然后第2格，最后第3格）。只输出数字顺序，不要其他解释。"
    },
    {
      "from": "gpt",
      "value": "0 2 3 1"
    }
  ],
  "images": ["ccac2025_complete/valid_zh_task1/216.jpg"],
  "metadata": {
    "comic_id": "216",
    "split": "valid",
    "language": "zh",
    "task": "task1"
  }
}
```

### 任务2示例

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<image>这是一组四格漫画，其中第2格的内容被遮罩。根据漫画图片和其他3格的描述，请生成第2格的文本描述。\n\n第0格描述：卡通猫（咪）正在一边哼着歌一边看电视...\n第1格描述：突然电视情节急转直下...\n第3格描述：卡通猫（咪）和卡通猪（猪仔）抱紧在一起...\n\n请输出生成的第2格描述："
    },
    {
      "from": "gpt",
      "value": "电视机出现"END"的字样，而卡通猫（咪）一脸虚弱的坐在沙发..."
    }
  ],
  "images": ["ccac2025_complete/valid_zh_task2/1200.jpg"],
  "metadata": {
    "comic_id": "1200",
    "mask_panel": 2,
    "split": "valid",
    "language": "zh",
    "task": "task2"
  }
}
```

---

## 在LlamaFactory中使用

### 1. WebUI配置

| 参数 | 值 |
|------|-----|
| `dataset_dir` | `./data` |
| `dataset` | 见下表 |

### 2. 可用的数据集名称（共18个）

**训练集（用于训练）**：
- `ccac2025_train_task1` / `_zh` / `_en`
- `ccac2025_train_task2` / `_zh` / `_en`

**验证集（用于评估，有标签）**：
- `ccac2025_valid_task1` / `_zh` / `_en`
- `ccac2025_valid_task2` / `_zh` / `_en`

**测试集（用于提交，无标签）**：
- `ccac2025_test_task1` / `_zh` / `_en`
- `ccac2025_test_task2` / `_zh` / `_en`

### 3. 推荐的多模态模型

- `Qwen2-VL` 系列（推荐）
- `LLaVA` 系列
- `InternVL2` 系列
- `MiniCPM-V` 系列

### 4. 训练配置建议

```yaml
# 基础配置
model_name_or_path: Qwen/Qwen2-VL-7B-Instruct
template: qwen2_vl

# 数据集配置
dataset_dir: ./data
dataset: ccac2025_train_task1  # 或 ccac2025_train_task2

# 训练参数
learning_rate: 2e-5
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
cutoff_len: 2048

# 评估配置（使用验证集）
val_size: 0.0  # 不使用自动划分，直接使用验证集
eval_dataset: ccac2025_valid_task1  # 评估时使用验证集
```

---

## 重要说明

### 1. 关于标签

| 数据集 | 标签情况 | 用途 |
|--------|----------|------|
| 训练集 | ✅ 有标签 | 训练模型 |
| 验证集 | ✅ 有标签 | 评估模型性能、调参 |
| 测试集 | ❌ 无标签 | 生成预测结果提交比赛 |

### 2. 关于编码

原始CSV文件使用 **GBK编码**，已正确处理转换为UTF-8。

### 3. 关于图片路径

JSON文件中的图片路径是相对于 `dataset_dir` 的相对路径：
```
ccac2025_complete/train_zh/xxx.jpg
ccac2025_complete/valid_zh_task1/xxx.jpg
ccac2025_complete/test_zh_task1/xxx.jpg
```

---

## 打包上传

如果需要将数据集上传到其他环境，建议打包整个文件夹：

```bash
cd /root/autodl-tmp/LlamaFactory/data
tar -czvf ccac2025_complete.tar.gz ccac2025_complete/
```

总大小约 **500MB**（包含所有图片）。

---

## 文件来源

原始数据来自CCAC2025比赛官方数据集：
- 训练集：`/root/autodl-tmp/dataset/CCAC2025manga_train/`
- 验证集：`/root/autodl-tmp/dataset/CCAC2025_valid/`
- 测试集：`/root/autodl-tmp/dataset/CCAC2025manga_test/`

---

## 构建脚本

如需重新构建数据集：

```bash
cd /root/autodl-tmp
python build_llamafactory_dataset_complete.py
```
