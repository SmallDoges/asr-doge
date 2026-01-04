# 🐕 ASR-Doge: Parameter-Efficient Speech Recognition with SmallDoge

<p align="center">
  <img src="https://img.shields.io/badge/WER-4.70%25-brightgreen" alt="WER">
  <img src="https://img.shields.io/badge/CER-2.75%25-brightgreen" alt="CER">
  <img src="https://img.shields.io/badge/RTF-0.91x-blue" alt="Real-Time Factor">
  <img src="https://img.shields.io/badge/Trainable_Params-0.05%25-orange" alt="Trainable Parameters">
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
</p>

**ASR-Doge** is a parameter-efficient automatic speech recognition (ASR) model that combines IBM's Granite Speech encoder with SmallDoge language model through a lightweight MLP adapter. This project demonstrates that competitive ASR performance can be achieved by training only **0.05%** of the total model parameters.

## 🎯 Key Results

| Metric | Score |
|--------|-------|
| **Word Error Rate (WER)** | 4.70% |
| **Character Error Rate (CER)** | 2.75% |
| **Perfect Match Rate** | 46.0% |
| **Real-Time Factor** | 0.91x (faster than real-time!) |
| **Trainable Parameters** | 1.57M / 3.36B (0.05%) |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ASR-Doge Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Audio Input (16kHz)                                         │
│       │                                                      │
│       ▼                                                      │
│  ┌────────────────────────────────────┐                      │
│  │   Speech Encoder (FROZEN)          │                      │
│  │   IBM Granite Speech 3.3-2B        │                      │
│  │   Parameters: ~3.04B               │                      │
│  └────────────────┬───────────────────┘                      │
│                   │ [B, T, 2048]                              │
│                   ▼                                           │
│  ┌────────────────────────────────────┐                      │
│  │   MLP Adapter (TRAINABLE)          │                      │
│  │   2048 → 512 → 1024                │                      │
│  │   Parameters: 1.57M                │                      │
│  └────────────────┬───────────────────┘                      │
│                   │ [B, T, 1024]                              │
│                   ▼                                           │
│  ┌────────────────────────────────────┐                      │
│  │   Language Model (TRAINABLE)       │                      │
│  │   SmallDoge-320M                   │                      │
│  │   Parameters: 320M                 │                      │
│  └────────────────┬───────────────────┘                      │
│                   │                                           │
│                   ▼                                           │
│            Text Transcription                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 24GB+ VRAM (40GB recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/SmallDoge/asr-doge.git
cd asr-doge

# Create conda environment
conda create -n asr-doge python=3.10
conda activate asr-doge

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
# requirements.txt
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.35.0
jiwer>=3.0.0
wandb>=0.15.0
tqdm>=4.65.0
numpy>=1.24.0
```

## 🚀 Quick Start

### 1. Download Dataset

```bash
# Download LibriSpeech (train-clean-100, dev-clean, test-clean)
python scripts/download_librispeech.py --output_dir ./data
```

### 2. Process Dataset

```bash
python src/data/data_processor.py \
    --dataset librispeech \
    --data_dir ./data \
    --output_dir ./processed \
    --train_split train-clean-100 \
    --dev_split dev-clean \
    --test_split test-clean
```

### 3. Train Model

```bash
python src/training/train.py \
    --data_dir ./processed \
    --output_dir ./checkpoints \
    --batch_size 16 \
    --learning_rate 2e-3 \
    --epochs 1 \
    --patience 2 \
    --wandb_project asr-doge
```

### 4. Run Benchmark

```bash
python src/benchmark/benchmark.py \
    --checkpoint_dir ./checkpoints/best \
    --data_dir ./processed \
    --output_dir ./benchmark_results \
    --max_samples 200
```

## 📊 Reproduce Our Results

To reproduce the exact results from our paper/TCC:

```bash
# Full training and evaluation pipeline
./scripts/reproduce.sh
```

Or step by step:

```bash
# 1. Process LibriSpeech data
python src/data/data_processor.py \
    --data_dir /path/to/librispeech \
    --output_dir ./processed

# 2. Train with our exact hyperparameters
python src/training/train.py \
    --data_dir ./processed \
    --output_dir ./checkpoints \
    --batch_size 16 \
    --learning_rate 0.002 \
    --epochs 1 \
    --patience 2 \
    --eval_steps 100 \
    --speech_encoder ibm-granite/granite-speech-3.3-2b \
    --language_model SmallDoge/Doge-320M-Checkpoint

# 3. Run comprehensive benchmark
python src/benchmark/benchmark.py \
    --checkpoint_dir ./checkpoints/best \
    --data_dir ./processed \
    --max_samples 200 \
    --num_speed_runs 10
```

## 📁 Project Structure

```
asr-doge/
├── README.md                     # This file
├── LICENSE                       # Apache 2.0 License
├── requirements.txt              # Python dependencies
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_processor.py     # Dataset processing utilities
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py              # Training script and model definition
│   ├── benchmark/
│   │   ├── __init__.py
│   │   └── benchmark.py          # Comprehensive benchmark script
│   ├── models/
│   │   ├── __init__.py
│   │   ├── configuration_singer_doge.py
│   │   └── modeling_singer_doge.py
│   └── modules/
│       ├── conv.py
│       ├── conv_norm.py
│       ├── ltsm.py
│       └── nae.py                # Neural Audio Encoder (SEANet)
├── scripts/
│   ├── download_librispeech.py   # Dataset download script
│   └── reproduce.sh              # Full reproduction script
├── configs/
│   └── default.yaml              # Default configuration
└── examples/
    └── inference.py              # Example inference script
```

## 🔧 Configuration

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `speech_encoder` | `ibm-granite/granite-speech-3.3-2b` | Pre-trained speech encoder |
| `language_model` | `SmallDoge/Doge-320M-Checkpoint` | Language model for text generation |
| `adapter_input_dim` | 2048 | Input dimension from speech encoder |
| `adapter_hidden_dim` | 512 | Hidden dimension in MLP adapter |
| `adapter_output_dim` | 1024 | Output dimension matching LM |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 16 | Training batch size |
| `learning_rate` | 2e-3 | Peak learning rate |
| `epochs` | 1 | Number of training epochs |
| `patience` | 2 | Early stopping patience |
| `eval_steps` | 100 | Steps between evaluations |

## 📈 Training Curves

Our model converges quickly due to the pre-trained components:

- **Training Loss**: 3.5 → 0.24 (over ~1,800 steps)
- **Validation CER**: Improves steadily with early stopping at patience=2

## 🧪 Evaluation

### Accuracy Metrics

```python
from src.benchmark import ASRDogeBenchmark, BenchmarkConfig

config = BenchmarkConfig(checkpoint_dir="./checkpoints/best")
benchmark = ASRDogeBenchmark(config)
benchmark.load_models()

# Run on test set
result = benchmark.run_full_benchmark(test_samples)
print(f"WER: {result.accuracy.wer:.2%}")
print(f"CER: {result.accuracy.cer:.2%}")
```

### Speed Benchmark

```python
# Measure inference speed
speed_result = benchmark.benchmark_speed(audio_path)
print(f"Latency: {speed_result['total_latency_ms']:.2f}ms")
print(f"Real-Time Factor: {speed_result['real_time_factor']:.2f}x")
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Areas for Contribution

1. **Model Improvements**: Better adapter architectures, LoRA integration
2. **Dataset Support**: Additional datasets beyond LibriSpeech
3. **Multilingual**: Leverage Granite's multilingual capabilities
4. **Streaming**: Real-time streaming ASR implementation
5. **Quantization**: INT8/INT4 quantization for edge deployment

## 📜 Citation

If you use ASR-Doge in your research, please cite:

```bibtex
@misc{asrdoge2026,
  title={ASR-Doge: Parameter-Efficient Speech Recognition with SmallDoge},
  author={Julio Hsu and SmallDoge Team},
  year={2026},
  howpublished={\url{https://github.com/SmallDoge/asr-doge}},
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SmallDoge Team** for the efficient Doge language model family
- **IBM** for the Granite Speech encoder
- **HuggingFace** for the transformers library
- **LibriSpeech** creators for the benchmark dataset

## 📬 Contact

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and community support
- **Email**: [your-email@example.com]

---

<p align="center">
  Made with ❤️ by the SmallDoge Team
</p>
