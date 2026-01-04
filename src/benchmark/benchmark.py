"""
ASR-Doge Benchmark Script
=========================
Comprehensive benchmarking for ASR-Doge model performance.

Measures:
- Accuracy: WER (Word Error Rate), CER (Character Error Rate)
- Speed: Latency, Tokens/second, Real-time factor
- Memory: Peak GPU memory usage

Usage:
    python benchmark.py \
        --checkpoint_dir ./checkpoints/best \
        --data_dir /path/to/librispeech \
        --output_dir ./benchmark_results
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import torchaudio
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from jiwer import wer, cer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("Warning: jiwer not installed. Install with: pip install jiwer")


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    # Model
    speech_encoder: str = "ibm-granite/granite-speech-3.3-2b"
    language_model: str = "SmallDoge/Doge-320M-Checkpoint"
    checkpoint_dir: str = "./checkpoints/best"
    
    # Adapter dimensions
    adapter_input_dim: int = 2048
    adapter_hidden_dim: int = 512
    adapter_output_dim: int = 1024
    
    # Benchmark settings
    batch_size: int = 1
    max_new_tokens: int = 200
    num_warmup_runs: int = 3
    num_speed_runs: int = 10
    max_samples: int = 200
    
    # Data
    data_dir: str = "./data"
    test_manifest: str = "test_manifest.json"
    
    # Output
    output_dir: str = "./benchmark_results"
    
    # Hardware
    device: str = "cuda"


@dataclass
class TimingResult:
    """Timing breakdown for a single inference"""
    audio_preprocessing_ms: float
    speech_embedding_ms: float
    adapter_projection_ms: float
    text_generation_ms: float
    total_inference_ms: float
    tokens_generated: int
    tokens_per_second: float


@dataclass 
class AccuracyResult:
    """Accuracy metrics for evaluation"""
    wer: float
    cer: float
    wer_std: float
    cer_std: float
    perfect_match_rate: float
    num_samples: int


@dataclass
class BenchmarkResult:
    """Complete benchmark results"""
    accuracy: AccuracyResult
    speed: Dict[str, float]
    memory: Dict[str, float]
    model_info: Dict[str, int]
    config: Dict
    timestamp: str
    samples: List[Dict]


class MLPAdapter(torch.nn.Module):
    """MLP Adapter for benchmark"""
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        output_dim: int = 1024
    ):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class ASRDogeBenchmark:
    """Benchmark runner for ASR-Doge"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Models will be loaded lazily
        self.speech_encoder = None
        self.speech_processor = None
        self.language_model = None
        self.tokenizer = None
        self.adapter = None
        
    def load_models(self):
        """Load all model components"""
        from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
        
        print("=" * 60)
        print("LOADING MODELS")
        print("=" * 60)
        
        # Speech encoder
        print(f"\n[1/4] Loading Speech Encoder: {self.config.speech_encoder}")
        self.speech_processor = AutoProcessor.from_pretrained(
            self.config.speech_encoder,
            trust_remote_code=True
        )
        self.speech_encoder = AutoModel.from_pretrained(
            self.config.speech_encoder,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(self.device).eval()
        print("✓ Speech encoder loaded")
        
        # Language model
        print(f"\n[2/4] Loading Language Model: {self.config.language_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.language_model,
            trust_remote_code=True
        )
        self.language_model = AutoModelForCausalLM.from_pretrained(
            self.config.language_model,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(self.device).eval()
        print("✓ Language model loaded")
        
        # Adapter
        print(f"\n[3/4] Creating MLP Adapter")
        self.adapter = MLPAdapter(
            input_dim=self.config.adapter_input_dim,
            hidden_dim=self.config.adapter_hidden_dim,
            output_dim=self.config.adapter_output_dim
        ).to(self.device).eval()
        print("✓ Adapter created")
        
        # Load checkpoints
        print(f"\n[4/4] Loading Checkpoints from: {self.config.checkpoint_dir}")
        adapter_path = os.path.join(self.config.checkpoint_dir, "adapter.pth")
        lm_path = os.path.join(self.config.checkpoint_dir, "lm.pth")
        
        # Also try alternative names
        if not os.path.exists(adapter_path):
            adapter_path = os.path.join(self.config.checkpoint_dir, "t_adaptor.pth")
        if not os.path.exists(lm_path):
            lm_path = os.path.join(self.config.checkpoint_dir, "lm.pth")
        
        if os.path.exists(adapter_path):
            state = torch.load(adapter_path, map_location=self.device)
            self.adapter.load_state_dict(state)
            print(f"✓ Loaded adapter from {adapter_path}")
        else:
            print(f"⚠ Adapter checkpoint not found at {adapter_path}")
        
        if os.path.exists(lm_path):
            state = torch.load(lm_path, map_location=self.device)
            self.language_model.load_state_dict(state, strict=False)
            print(f"✓ Loaded LM from {lm_path}")
        else:
            print(f"⚠ LM checkpoint not found at {lm_path}")
        
        print("\n" + "=" * 60)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters"""
        speech_params = sum(p.numel() for p in self.speech_encoder.parameters())
        adapter_params = sum(p.numel() for p in self.adapter.parameters())
        lm_params = sum(p.numel() for p in self.language_model.parameters())
        total = speech_params + adapter_params + lm_params
        trainable = adapter_params + lm_params  # Speech encoder is frozen
        
        return {
            "total": total,
            "trainable": trainable,
            "speech_encoder": speech_params,
            "adapter": adapter_params,
            "language_model": lm_params
        }
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0)
    
    def _sync_device(self):
        """Synchronize CUDA device for accurate timing"""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
    
    @torch.no_grad()
    def run_inference(
        self,
        audio: torch.Tensor,
        instruction: str = "transcribe the following audio: "
    ) -> str:
        """Run full inference pipeline"""
        audio = audio.unsqueeze(0).to(self.device)
        
        # Process through speech processor
        audio_features = self.speech_processor(
            audio.squeeze(0).cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Get speech embeddings
        speech_embeddings = self.speech_encoder.get_audio_features(audio_features)
        projected = self.adapter(speech_embeddings)
        
        # Get instruction embeddings
        instruction_ids = self.tokenizer(
            instruction.lower(),
            return_tensors="pt"
        ).input_ids.to(self.device)
        instruction_emb = self.language_model.get_input_embeddings()(instruction_ids)
        
        # Combine embeddings
        current_emb = torch.cat([instruction_emb, projected], dim=1)
        
        # Generate
        generated_ids = []
        for _ in range(self.config.max_new_tokens):
            outputs = self.language_model(inputs_embeds=current_emb)
            next_logits = outputs.logits[:, -1, :]
            next_id = torch.argmax(next_logits, dim=-1)
            
            if next_id.item() == self.tokenizer.eos_token_id:
                break
            
            generated_ids.append(next_id.item())
            next_emb = self.language_model.get_input_embeddings()(next_id.unsqueeze(0))
            current_emb = torch.cat([current_emb, next_emb], dim=1)
        
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
    
    @torch.no_grad()
    def run_timed_inference(self, audio: torch.Tensor) -> Tuple[str, TimingResult]:
        """Run inference with detailed timing"""
        audio = audio.unsqueeze(0).to(self.device)
        instruction = "transcribe the following audio: "
        
        # 1. Audio preprocessing
        self._sync_device()
        t0 = time.perf_counter()
        audio_features = self.speech_processor(
            audio.squeeze(0).cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device)
        self._sync_device()
        t1 = time.perf_counter()
        audio_preprocessing_ms = (t1 - t0) * 1000
        
        # 2. Speech embedding
        self._sync_device()
        t2 = time.perf_counter()
        speech_embeddings = self.speech_encoder.get_audio_features(audio_features)
        self._sync_device()
        t3 = time.perf_counter()
        speech_embedding_ms = (t3 - t2) * 1000
        
        # 3. Adapter projection
        self._sync_device()
        t4 = time.perf_counter()
        projected = self.adapter(speech_embeddings)
        instruction_ids = self.tokenizer(
            instruction.lower(),
            return_tensors="pt"
        ).input_ids.to(self.device)
        instruction_emb = self.language_model.get_input_embeddings()(instruction_ids)
        current_emb = torch.cat([instruction_emb, projected], dim=1)
        self._sync_device()
        t5 = time.perf_counter()
        adapter_projection_ms = (t5 - t4) * 1000
        
        # 4. Text generation
        self._sync_device()
        t6 = time.perf_counter()
        generated_ids = []
        for _ in range(self.config.max_new_tokens):
            outputs = self.language_model(inputs_embeds=current_emb)
            next_logits = outputs.logits[:, -1, :]
            next_id = torch.argmax(next_logits, dim=-1)
            
            if next_id.item() == self.tokenizer.eos_token_id:
                break
            
            generated_ids.append(next_id.item())
            next_emb = self.language_model.get_input_embeddings()(next_id.unsqueeze(0))
            current_emb = torch.cat([current_emb, next_emb], dim=1)
        self._sync_device()
        t7 = time.perf_counter()
        text_generation_ms = (t7 - t6) * 1000
        
        total_ms = audio_preprocessing_ms + speech_embedding_ms + adapter_projection_ms + text_generation_ms
        tokens = len(generated_ids)
        tps = (tokens / text_generation_ms * 1000) if text_generation_ms > 0 else 0
        
        transcription = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
        
        timing = TimingResult(
            audio_preprocessing_ms=audio_preprocessing_ms,
            speech_embedding_ms=speech_embedding_ms,
            adapter_projection_ms=adapter_projection_ms,
            text_generation_ms=text_generation_ms,
            total_inference_ms=total_ms,
            tokens_generated=tokens,
            tokens_per_second=tps
        )
        
        return transcription, timing
    
    def benchmark_accuracy(self, samples: List[Dict]) -> Tuple[AccuracyResult, List[Dict]]:
        """Benchmark accuracy on test samples"""
        print("\n" + "=" * 60)
        print("ACCURACY BENCHMARK")
        print("=" * 60)
        
        predictions = []
        references = []
        sample_results = []
        
        for sample in tqdm(samples[:self.config.max_samples], desc="Evaluating"):
            audio = self.preprocess_audio(sample["audio_path"])
            prediction = self.run_inference(audio)
            reference = sample["text"].lower().strip()
            
            predictions.append(prediction)
            references.append(reference)
            
            # Per-sample metrics
            if JIWER_AVAILABLE and prediction and reference:
                sample_wer = wer([reference], [prediction])
                sample_cer = cer([reference], [prediction])
            else:
                sample_wer = 1.0
                sample_cer = 1.0
            
            sample_results.append({
                "id": sample["id"],
                "reference": reference,
                "prediction": prediction,
                "wer": sample_wer,
                "cer": sample_cer,
                "perfect_match": prediction == reference
            })
        
        # Aggregate metrics
        if JIWER_AVAILABLE:
            # Filter empty
            valid = [(p, r) for p, r in zip(predictions, references) if p and r]
            if valid:
                preds, refs = zip(*valid)
                total_wer = wer(list(refs), list(preds))
                total_cer = cer(list(refs), list(preds))
            else:
                total_wer = 1.0
                total_cer = 1.0
        else:
            total_wer = 0.0
            total_cer = 0.0
        
        # Statistics
        import numpy as np
        wers = [s["wer"] for s in sample_results]
        cers = [s["cer"] for s in sample_results]
        perfect = sum(1 for s in sample_results if s["perfect_match"])
        
        result = AccuracyResult(
            wer=total_wer,
            cer=total_cer,
            wer_std=float(np.std(wers)),
            cer_std=float(np.std(cers)),
            perfect_match_rate=perfect / len(sample_results),
            num_samples=len(sample_results)
        )
        
        print(f"\n📊 Accuracy Results:")
        print(f"   WER: {result.wer*100:.2f}% (±{result.wer_std*100:.2f}%)")
        print(f"   CER: {result.cer*100:.2f}% (±{result.cer_std*100:.2f}%)")
        print(f"   Perfect Match: {result.perfect_match_rate*100:.1f}%")
        print(f"   Samples: {result.num_samples}")
        
        return result, sample_results
    
    def benchmark_speed(self, sample_audio: str) -> Dict[str, float]:
        """Benchmark inference speed"""
        print("\n" + "=" * 60)
        print("SPEED BENCHMARK")
        print("=" * 60)
        
        audio = self.preprocess_audio(sample_audio)
        audio_duration = len(audio) / 16000
        
        # Warmup
        print(f"\nWarming up ({self.config.num_warmup_runs} runs)...")
        for _ in range(self.config.num_warmup_runs):
            self.run_inference(audio)
        
        # Timed runs
        print(f"Running benchmark ({self.config.num_speed_runs} runs)...")
        timings = []
        for _ in tqdm(range(self.config.num_speed_runs)):
            _, timing = self.run_timed_inference(audio)
            timings.append(timing)
        
        # Aggregate
        import numpy as np
        avg_total = np.mean([t.total_inference_ms for t in timings])
        avg_audio = np.mean([t.audio_preprocessing_ms for t in timings])
        avg_speech = np.mean([t.speech_embedding_ms for t in timings])
        avg_adapter = np.mean([t.adapter_projection_ms for t in timings])
        avg_gen = np.mean([t.text_generation_ms for t in timings])
        avg_tokens = np.mean([t.tokens_generated for t in timings])
        avg_tps = np.mean([t.tokens_per_second for t in timings])
        
        rtf = (avg_total / 1000) / audio_duration
        
        result = {
            "audio_duration_s": audio_duration,
            "total_latency_ms": avg_total,
            "audio_preprocessing_ms": avg_audio,
            "speech_embedding_ms": avg_speech,
            "adapter_projection_ms": avg_adapter,
            "text_generation_ms": avg_gen,
            "tokens_generated": avg_tokens,
            "tokens_per_second": avg_tps,
            "real_time_factor": rtf,
            "is_real_time": rtf < 1.0
        }
        
        print(f"\n⚡ Speed Results:")
        print(f"   Audio Duration: {audio_duration:.2f}s")
        print(f"   Total Latency: {avg_total:.2f}ms")
        print(f"   Real-Time Factor: {rtf:.2f}x")
        print(f"   Tokens/Second: {avg_tps:.1f}")
        print(f"   Real-Time Capable: {'✓ Yes' if rtf < 1.0 else '✗ No'}")
        print(f"\n   Breakdown:")
        print(f"   - Audio Preprocessing: {avg_audio:.2f}ms ({avg_audio/avg_total*100:.1f}%)")
        print(f"   - Speech Embedding: {avg_speech:.2f}ms ({avg_speech/avg_total*100:.1f}%)")
        print(f"   - Adapter Projection: {avg_adapter:.2f}ms ({avg_adapter/avg_total*100:.1f}%)")
        print(f"   - Text Generation: {avg_gen:.2f}ms ({avg_gen/avg_total*100:.1f}%)")
        
        return result
    
    def benchmark_memory(self) -> Dict[str, float]:
        """Measure GPU memory usage"""
        print("\n" + "=" * 60)
        print("MEMORY BENCHMARK")
        print("=" * 60)
        
        if not torch.cuda.is_available():
            print("CUDA not available, skipping memory benchmark")
            return {"peak_memory_mb": 0, "allocated_memory_mb": 0}
        
        torch.cuda.reset_peak_memory_stats()
        
        # Run a sample inference
        dummy_audio = torch.randn(16000 * 5)  # 5 seconds
        self.run_inference(dummy_audio)
        
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        
        result = {
            "peak_memory_mb": peak,
            "allocated_memory_mb": allocated
        }
        
        print(f"\n💾 Memory Results:")
        print(f"   Peak Memory: {peak:.1f} MB")
        print(f"   Allocated Memory: {allocated:.1f} MB")
        
        return result
    
    def run_full_benchmark(self, samples: List[Dict]) -> BenchmarkResult:
        """Run complete benchmark suite"""
        print("\n" + "=" * 60)
        print("ASR-DOGE COMPREHENSIVE BENCHMARK")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Samples: {min(len(samples), self.config.max_samples)}")
        print("=" * 60)
        
        # Load models
        self.load_models()
        
        # Model info
        model_info = self.count_parameters()
        print(f"\n🏗️ Model Info:")
        print(f"   Total Parameters: {model_info['total']/1e9:.2f}B")
        print(f"   Trainable Parameters: {model_info['trainable']/1e6:.2f}M")
        print(f"   Adapter Parameters: {model_info['adapter']/1e6:.2f}M")
        
        # Run benchmarks
        accuracy, sample_results = self.benchmark_accuracy(samples)
        speed = self.benchmark_speed(samples[0]["audio_path"])
        memory = self.benchmark_memory()
        
        # Create result
        result = BenchmarkResult(
            accuracy=accuracy,
            speed=speed,
            memory=memory,
            model_info=model_info,
            config=asdict(self.config),
            timestamp=datetime.now().isoformat(),
            samples=sample_results[:10]  # Save first 10 samples for reference
        )
        
        return result
    
    def save_results(self, result: BenchmarkResult, output_path: str):
        """Save benchmark results to JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert dataclasses to dicts
        output = {
            "accuracy": asdict(result.accuracy),
            "speed": result.speed,
            "memory": result.memory,
            "model_info": result.model_info,
            "config": result.config,
            "timestamp": result.timestamp,
            "samples": result.samples
        }
        
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")


def load_test_samples(manifest_path: str) -> List[Dict]:
    """Load test samples from manifest"""
    samples = []
    with open(manifest_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Benchmark ASR-Doge model")
    
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing model checkpoints")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing test data")
    parser.add_argument("--test_manifest", type=str, default="test_manifest.json",
                        help="Test manifest filename")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                        help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Maximum samples for accuracy benchmark")
    parser.add_argument("--num_speed_runs", type=int, default=10,
                        help="Number of runs for speed benchmark")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    
    # Model arguments
    parser.add_argument("--speech_encoder", type=str,
                        default="ibm-granite/granite-speech-3.3-2b")
    parser.add_argument("--language_model", type=str,
                        default="SmallDoge/Doge-320M-Checkpoint")
    
    args = parser.parse_args()
    
    # Create config
    config = BenchmarkConfig(
        speech_encoder=args.speech_encoder,
        language_model=args.language_model,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        test_manifest=args.test_manifest,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_speed_runs=args.num_speed_runs,
        device=args.device
    )
    
    # Load test samples
    manifest_path = os.path.join(args.data_dir, args.test_manifest)
    print(f"Loading test samples from {manifest_path}")
    samples = load_test_samples(manifest_path)
    print(f"Loaded {len(samples)} samples")
    
    # Run benchmark
    benchmark = ASRDogeBenchmark(config)
    result = benchmark.run_full_benchmark(samples)
    
    # Save results
    output_path = os.path.join(
        args.output_dir,
        f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    benchmark.save_results(result, output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"WER: {result.accuracy.wer*100:.2f}%")
    print(f"CER: {result.accuracy.cer*100:.2f}%")
    print(f"Real-Time Factor: {result.speed['real_time_factor']:.2f}x")
    print(f"Tokens/Second: {result.speed['tokens_per_second']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

