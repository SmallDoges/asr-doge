#!/bin/bash
# ASR-Doge: Full Reproduction Script
# This script reproduces the exact results from our TCC/paper
#
# Usage: ./scripts/reproduce.sh /path/to/librispeech
#
# Expected Results:
#   - WER: ~4.70%
#   - CER: ~2.75%
#   - RTF: ~0.91x

set -e

# Configuration
DATA_DIR="${1:-./data}"
OUTPUT_DIR="./experiments/reproduction_$(date +%Y%m%d_%H%M%S)"
PROCESSED_DIR="$OUTPUT_DIR/processed"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
BENCHMARK_DIR="$OUTPUT_DIR/benchmark"

echo "=============================================="
echo "ASR-Doge Reproduction Script"
echo "=============================================="
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "=============================================="

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROCESSED_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$BENCHMARK_DIR"

# Step 1: Process Dataset
echo ""
echo "[1/3] Processing Dataset..."
echo "----------------------------------------------"
python src/data/data_processor.py \
    --dataset librispeech \
    --data_dir "$DATA_DIR" \
    --output_dir "$PROCESSED_DIR" \
    --train_split train-clean-100 \
    --dev_split dev-clean \
    --test_split test-clean \
    --sample_rate 16000 \
    --max_duration 30.0 \
    --min_duration 0.5

# Step 2: Train Model
echo ""
echo "[2/3] Training Model..."
echo "----------------------------------------------"
python src/training/train.py \
    --data_dir "$PROCESSED_DIR" \
    --train_manifest train_manifest.json \
    --dev_manifest dev_manifest.json \
    --output_dir "$CHECKPOINT_DIR" \
    --speech_encoder ibm-granite/granite-speech-3.3-2b \
    --language_model SmallDoge/Doge-320M-Checkpoint \
    --adapter_hidden_dim 512 \
    --batch_size 16 \
    --learning_rate 0.002 \
    --epochs 1 \
    --patience 2 \
    --eval_steps 100 \
    --save_steps 500 \
    --device cuda \
    --wandb_project asr-doge-reproduction \
    --wandb_run_name "reproduction_$(date +%Y%m%d_%H%M%S)"

# Step 3: Run Benchmark
echo ""
echo "[3/3] Running Benchmark..."
echo "----------------------------------------------"
python src/benchmark/benchmark.py \
    --checkpoint_dir "$CHECKPOINT_DIR/best" \
    --data_dir "$PROCESSED_DIR" \
    --test_manifest test_manifest.json \
    --output_dir "$BENCHMARK_DIR" \
    --max_samples 200 \
    --num_speed_runs 10 \
    --device cuda \
    --speech_encoder ibm-granite/granite-speech-3.3-2b \
    --language_model SmallDoge/Doge-320M-Checkpoint

# Summary
echo ""
echo "=============================================="
echo "REPRODUCTION COMPLETE"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Benchmark results: $BENCHMARK_DIR"
echo "Model checkpoints: $CHECKPOINT_DIR"
echo ""
echo "Expected Results:"
echo "  WER: ~4.70%"
echo "  CER: ~2.75%"
echo "  RTF: ~0.91x"
echo "=============================================="

