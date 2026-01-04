"""
ASR-Doge Data Processor
=======================
Dataset processing utilities for LibriSpeech and custom audio datasets.

This module handles:
- Audio loading and preprocessing
- Text normalization
- Dataset splitting and batching
- Data augmentation (optional)

Usage:
    python data_processor.py --dataset librispeech --split train-clean-100 --output_dir ./processed
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


@dataclass
class AudioConfig:
    """Configuration for audio preprocessing"""
    sample_rate: int = 16000
    max_duration: float = 30.0  # seconds
    min_duration: float = 0.5   # seconds
    normalize: bool = True
    

@dataclass
class DataConfig:
    """Configuration for dataset processing"""
    dataset_name: str = "librispeech"
    data_dir: str = "./data"
    output_dir: str = "./processed"
    train_split: str = "train-clean-100"
    dev_split: str = "dev-clean"
    test_split: str = "test-clean"
    num_workers: int = 4
    audio_config: AudioConfig = field(default_factory=AudioConfig)


class TextProcessor:
    """Text normalization and cleaning utilities"""
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for ASR training"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except apostrophes
        import re
        text = re.sub(r"[^\w\s']", "", text)
        
        # Normalize whitespace
        text = " ".join(text.split())
        
        return text.strip()
    
    @staticmethod
    def clean_for_metrics(text: str) -> str:
        """Clean text for WER/CER computation"""
        text = text.lower()
        # Remove all punctuation
        import re
        text = re.sub(r"[^\w\s]", "", text)
        text = " ".join(text.split())
        return text.strip()


class AudioProcessor:
    """Audio loading and preprocessing utilities"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        
    def load_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file and return waveform and sample rate"""
        waveform, sr = torchaudio.load(path)
        return waveform, sr
    
    def resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate"""
        if orig_sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.config.sample_rate)
            waveform = resampler(waveform)
        return waveform
    
    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio waveform"""
        if self.config.normalize:
            waveform = waveform / (waveform.abs().max() + 1e-8)
        return waveform
    
    def process(self, path: str) -> Optional[torch.Tensor]:
        """Full audio processing pipeline"""
        try:
            waveform, sr = self.load_audio(path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample
            waveform = self.resample(waveform, sr)
            
            # Check duration
            duration = waveform.shape[1] / self.config.sample_rate
            if duration < self.config.min_duration or duration > self.config.max_duration:
                return None
            
            # Normalize
            waveform = self.normalize_audio(waveform)
            
            return waveform.squeeze(0)  # Return 1D tensor
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return None


class LibriSpeechProcessor:
    """Process LibriSpeech dataset"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.audio_processor = AudioProcessor(config.audio_config)
        self.text_processor = TextProcessor()
        
    def find_audio_files(self, split_dir: Path) -> List[Dict]:
        """Find all audio files and their transcriptions"""
        samples = []
        
        for trans_file in split_dir.rglob("*.trans.txt"):
            # Read transcription file
            with open(trans_file, "r") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) != 2:
                        continue
                    
                    audio_id, text = parts
                    
                    # Find corresponding audio file
                    audio_path = trans_file.parent / f"{audio_id}.flac"
                    if not audio_path.exists():
                        continue
                    
                    samples.append({
                        "id": audio_id,
                        "audio_path": str(audio_path),
                        "text": self.text_processor.normalize(text),
                        "raw_text": text
                    })
        
        return samples
    
    def process_split(self, split: str) -> List[Dict]:
        """Process a single dataset split"""
        split_dir = Path(self.config.data_dir) / "LibriSpeech" / split
        
        if not split_dir.exists():
            # Try alternate path structure
            split_dir = Path(self.config.data_dir) / split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        print(f"Processing {split}...")
        samples = self.find_audio_files(split_dir)
        
        # Process each sample
        processed_samples = []
        for sample in tqdm(samples, desc=f"Processing {split}"):
            waveform = self.audio_processor.process(sample["audio_path"])
            if waveform is not None:
                sample["duration"] = len(waveform) / self.config.audio_config.sample_rate
                processed_samples.append(sample)
        
        print(f"Processed {len(processed_samples)}/{len(samples)} samples from {split}")
        return processed_samples
    
    def process_all(self) -> Dict[str, List[Dict]]:
        """Process all splits"""
        results = {}
        
        for split_name, split_key in [
            ("train", self.config.train_split),
            ("dev", self.config.dev_split),
            ("test", self.config.test_split)
        ]:
            try:
                results[split_name] = self.process_split(split_key)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                results[split_name] = []
        
        return results
    
    def save_manifest(self, samples: List[Dict], output_path: str):
        """Save processed samples to JSON manifest"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        
        print(f"Saved {len(samples)} samples to {output_path}")


class ASRDogeDataset(Dataset):
    """PyTorch Dataset for ASR-Doge training"""
    
    def __init__(
        self,
        manifest_path: str,
        audio_processor: AudioProcessor,
        tokenizer=None,
        max_samples: Optional[int] = None
    ):
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer
        
        # Load manifest
        self.samples = []
        with open(manifest_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))
        
        if max_samples:
            self.samples = self.samples[:max_samples]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load and process audio
        waveform = self.audio_processor.process(sample["audio_path"])
        if waveform is None:
            # Return empty tensor for failed samples
            waveform = torch.zeros(16000)  # 1 second of silence
        
        result = {
            "id": sample["id"],
            "audio": waveform,
            "text": sample["text"],
            "duration": sample.get("duration", len(waveform) / 16000)
        }
        
        # Tokenize if tokenizer is provided
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                sample["text"],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=512
            )
            result["input_ids"] = tokens.input_ids.squeeze(0)
            result["attention_mask"] = tokens.attention_mask.squeeze(0)
        
        return result


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching"""
    # Separate audio and text
    audios = [item["audio"] for item in batch]
    texts = [item["text"] for item in batch]
    ids = [item["id"] for item in batch]
    durations = [item["duration"] for item in batch]
    
    # Pad audio to same length
    max_len = max(len(a) for a in audios)
    padded_audios = torch.zeros(len(audios), max_len)
    audio_lengths = []
    
    for i, audio in enumerate(audios):
        length = len(audio)
        padded_audios[i, :length] = audio
        audio_lengths.append(length)
    
    result = {
        "id": ids,
        "audio": padded_audios,
        "audio_lengths": torch.tensor(audio_lengths),
        "text": texts,
        "duration": durations
    }
    
    # Handle tokenized inputs if present
    if "input_ids" in batch[0]:
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        
        # Pad sequences
        max_seq_len = max(len(ids) for ids in input_ids)
        padded_ids = torch.zeros(len(input_ids), max_seq_len, dtype=torch.long)
        padded_masks = torch.zeros(len(attention_masks), max_seq_len, dtype=torch.long)
        
        for i, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
            padded_ids[i, :len(ids)] = ids
            padded_masks[i, :len(mask)] = mask
        
        result["input_ids"] = padded_ids
        result["attention_mask"] = padded_masks
    
    return result


def create_dataloaders(
    train_manifest: str,
    dev_manifest: str,
    test_manifest: str,
    audio_config: AudioConfig,
    batch_size: int = 16,
    num_workers: int = 4,
    tokenizer=None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for training, validation, and testing"""
    
    audio_processor = AudioProcessor(audio_config)
    
    train_dataset = ASRDogeDataset(train_manifest, audio_processor, tokenizer)
    dev_dataset = ASRDogeDataset(dev_manifest, audio_processor, tokenizer)
    test_dataset = ASRDogeDataset(test_manifest, audio_processor, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, dev_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Process datasets for ASR-Doge")
    parser.add_argument("--dataset", type=str, default="librispeech",
                        help="Dataset name (librispeech)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to raw dataset")
    parser.add_argument("--output_dir", type=str, default="./processed",
                        help="Output directory for processed data")
    parser.add_argument("--train_split", type=str, default="train-clean-100",
                        help="Training split name")
    parser.add_argument("--dev_split", type=str, default="dev-clean",
                        help="Validation split name")
    parser.add_argument("--test_split", type=str, default="test-clean",
                        help="Test split name")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Target sample rate")
    parser.add_argument("--max_duration", type=float, default=30.0,
                        help="Maximum audio duration in seconds")
    parser.add_argument("--min_duration", type=float, default=0.5,
                        help="Minimum audio duration in seconds")
    
    args = parser.parse_args()
    
    # Create config
    audio_config = AudioConfig(
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        min_duration=args.min_duration
    )
    
    config = DataConfig(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        dev_split=args.dev_split,
        test_split=args.test_split,
        audio_config=audio_config
    )
    
    # Process dataset
    if args.dataset.lower() == "librispeech":
        processor = LibriSpeechProcessor(config)
        results = processor.process_all()
        
        # Save manifests
        for split_name, samples in results.items():
            if samples:
                output_path = os.path.join(args.output_dir, f"{split_name}_manifest.json")
                processor.save_manifest(samples, output_path)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    print("\n✅ Dataset processing complete!")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

