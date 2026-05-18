#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/autodl-tmp/LlamaFactory"
cd "$ROOT"
python data/ccac2025_complete/task2/build_task2_push_dataset.py --zh_repeat "${1:-5}"
echo ">>> llamafactory-cli train (GPU 占用高，建议 screen/tmux)"
llamafactory-cli train "$ROOT/train_task2_push_baseline.yaml"
echo ">>> 训练结束，请执行："
echo "cd /root/autodl-tmp && python evaluate_and_test/pick_best_task2_ckpt.py --run_dir $ROOT/saves/qwen2.5-vl-lora-task2-push"
