"""
ASR-Doge Inference Example
==========================
Simple example demonstrating how to use ASR-Doge for speech transcription.

Usage:
    python examples/inference.py \
        --checkpoint_dir ./checkpoints/best \
        --audio_path /path/to/audio.wav
"""

import sys
import argparse
from pathlib import Path

import torch
import torchaudio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.train import ASRDogeModel, ModelConfig


def load_model(checkpoint_dir: str, device: str = "cuda") -> ASRDogeModel:
    """Load trained ASR-Doge model"""
    config = ModelConfig(
        speech_encoder="ibm-granite/granite-speech-3.3-2b",
        language_model="SmallDoge/Doge-320M-Checkpoint",
        adapter_hidden_dim=512
    )
    
    model = ASRDogeModel(config, device=device)
    
    # Load checkpoints
    adapter_path = Path(checkpoint_dir) / "adapter.pth"
    lm_path = Path(checkpoint_dir) / "lm.pth"
    
    if adapter_path.exists():
        model.adapter.load_state_dict(torch.load(adapter_path, map_location=device))
        print(f"✓ Loaded adapter from {adapter_path}")
    
    if lm_path.exists():
        model.language_model.load_state_dict(torch.load(lm_path, map_location=device), strict=False)
        print(f"✓ Loaded LM from {lm_path}")
    
    model.eval()
    return model


def transcribe_audio(
    model: ASRDogeModel,
    audio_path: str,
    device: str = "cuda"
) -> str:
    """Transcribe a single audio file"""
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample to 16kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    
    # Process through speech processor
    audio_features = model.speech_processor(
        waveform.squeeze(0).numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)
    
    # Generate transcription
    transcriptions = model.generate(audio_features)
    
    return transcriptions[0]


def main():
    parser = argparse.ArgumentParser(description="ASR-Doge Inference")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing model checkpoints")
    parser.add_argument("--audio_path", type=str, required=True,
                        help="Path to audio file to transcribe")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("ASR-Doge Inference")
    print("=" * 50)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint_dir, args.device)
    
    # Transcribe
    print(f"\nTranscribing: {args.audio_path}")
    transcription = transcribe_audio(model, args.audio_path, args.device)
    
    print("\n" + "=" * 50)
    print("TRANSCRIPTION:")
    print("=" * 50)
    print(transcription)
    print("=" * 50)


if __name__ == "__main__":
    main()

