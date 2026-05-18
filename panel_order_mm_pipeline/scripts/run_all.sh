#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Step 1: Preparing data (split panels + extract ViT features)..."
python src/prepare_data.py

echo "Step 2: Training multimodal temporal model..."
python src/train.py

echo "Step 3: Evaluating..."
python src/evaluate.py

echo "All done! Check outputs/ for results."
