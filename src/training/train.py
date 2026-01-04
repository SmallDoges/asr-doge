"""
ASR-Doge Training Script
========================
End-to-end training pipeline for the ASR-Doge speech recognition model.

Architecture:
    - Speech Encoder: IBM Granite Speech 3.3-2B (frozen)
    - Adapter: 2-layer MLP (trainable)
    - Language Model: SmallDoge-320M (trainable)

Usage:
    python train.py --config configs/default.yaml
    
    # Or with command line arguments:
    python train.py \
        --data_dir /path/to/librispeech \
        --output_dir ./checkpoints \
        --batch_size 16 \
        --learning_rate 2e-3 \
        --epochs 1
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging to wandb disabled.")

try:
    from jiwer import wer, cer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("Warning: jiwer not installed. WER/CER metrics disabled.")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Speech encoder
    speech_encoder: str = "ibm-granite/granite-speech-3.3-2b"
    freeze_speech_encoder: bool = True
    
    # Adapter
    adapter_input_dim: int = 2048
    adapter_hidden_dim: int = 512
    adapter_output_dim: int = 1024
    adapter_activation: str = "gelu"
    
    # Language model
    language_model: str = "SmallDoge/Doge-320M-Checkpoint"
    freeze_language_model: bool = False


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    data_dir: str = "./data"
    train_manifest: str = "train_manifest.json"
    dev_manifest: str = "dev_manifest.json"
    test_manifest: str = "test_manifest.json"
    
    # Training
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    epochs: int = 1
    
    # Evaluation
    eval_steps: int = 100
    eval_metrics_freq: int = 500
    max_eval_samples: int = 200
    
    # Checkpointing
    output_dir: str = "./checkpoints"
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Early stopping
    early_stopping_patience: int = 2
    early_stopping_metric: str = "cer"
    
    # Hardware
    device: str = "cuda"
    fp16: bool = False  # Use FP32 for stability
    num_workers: int = 4
    
    # Logging
    wandb_project: str = "asr-doge"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_steps: int = 10


class MLPAdapter(nn.Module):
    """MLP Adapter to bridge speech encoder and language model dimensions"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.adapter_input_dim, config.adapter_hidden_dim)
        
        if config.adapter_activation == "gelu":
            self.activation = nn.GELU()
        elif config.adapter_activation == "silu":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        self.fc2 = nn.Linear(config.adapter_hidden_dim, config.adapter_output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class ASRDogeModel(nn.Module):
    """
    ASR-Doge: Speech-Aware Text Model
    
    Combines a frozen speech encoder with a trainable adapter and language model
    for efficient speech-to-text transcription.
    """
    
    def __init__(self, model_config: ModelConfig, device: str = "cuda"):
        super().__init__()
        self.config = model_config
        self.device = device
        
        # Load components
        self._load_speech_encoder()
        self._load_language_model()
        self._create_adapter()
        
    def _load_speech_encoder(self):
        """Load and optionally freeze speech encoder"""
        from transformers import AutoModel, AutoProcessor
        
        logger.info(f"Loading speech encoder: {self.config.speech_encoder}")
        self.speech_processor = AutoProcessor.from_pretrained(
            self.config.speech_encoder,
            trust_remote_code=True
        )
        self.speech_encoder = AutoModel.from_pretrained(
            self.config.speech_encoder,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(self.device)
        
        if self.config.freeze_speech_encoder:
            for param in self.speech_encoder.parameters():
                param.requires_grad = False
            self.speech_encoder.eval()
            logger.info("Speech encoder frozen")
    
    def _load_language_model(self):
        """Load language model"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading language model: {self.config.language_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.language_model,
            trust_remote_code=True
        )
        self.language_model = AutoModelForCausalLM.from_pretrained(
            self.config.language_model,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(self.device)
        
        if self.config.freeze_language_model:
            for param in self.language_model.parameters():
                param.requires_grad = False
            logger.info("Language model frozen")
    
    def _create_adapter(self):
        """Create MLP adapter"""
        logger.info("Creating MLP adapter")
        self.adapter = MLPAdapter(self.config).to(self.device)
    
    def get_speech_embeddings(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Extract speech embeddings from audio features"""
        with torch.no_grad() if self.config.freeze_speech_encoder else torch.enable_grad():
            speech_embeddings = self.speech_encoder.get_audio_features(audio_features)
        return speech_embeddings
    
    def forward(
        self,
        audio_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        instruction: str = "transcribe the following audio: "
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training
        
        Args:
            audio_features: Preprocessed audio features
            labels: Target token IDs for teacher forcing
            instruction: Instruction prefix
            
        Returns:
            Dictionary containing loss and logits
        """
        # Get speech embeddings
        speech_embeddings = self.get_speech_embeddings(audio_features)
        
        # Project through adapter
        projected_embeddings = self.adapter(speech_embeddings)
        
        # Get instruction embeddings
        instruction_ids = self.tokenizer(
            instruction.lower(),
            return_tensors="pt",
            padding=False
        ).input_ids.to(self.device)
        instruction_embeddings = self.language_model.get_input_embeddings()(instruction_ids)
        
        # Combine embeddings
        if labels is not None:
            # Teacher forcing: include label embeddings
            label_embeddings = self.language_model.get_input_embeddings()(labels)
            combined_embeddings = torch.cat([
                instruction_embeddings.expand(projected_embeddings.size(0), -1, -1),
                projected_embeddings,
                label_embeddings
            ], dim=1)
        else:
            combined_embeddings = torch.cat([
                instruction_embeddings.expand(projected_embeddings.size(0), -1, -1),
                projected_embeddings
            ], dim=1)
        
        # Forward through language model
        outputs = self.language_model(inputs_embeds=combined_embeddings)
        logits = outputs.logits
        
        result = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            # Get the portion of logits that corresponds to labels
            instruction_len = instruction_embeddings.size(1)
            speech_len = projected_embeddings.size(1)
            prefix_len = instruction_len + speech_len
            
            # Shift for causal LM loss
            shift_logits = logits[:, prefix_len-1:-1, :].contiguous()
            shift_labels = labels.contiguous()
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            result["loss"] = loss
        
        return result
    
    def generate(
        self,
        audio_features: torch.Tensor,
        instruction: str = "transcribe the following audio: ",
        max_new_tokens: int = 200
    ) -> List[str]:
        """
        Generate transcriptions for audio inputs
        
        Args:
            audio_features: Preprocessed audio features
            instruction: Instruction prefix
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of transcription strings
        """
        self.eval()
        
        with torch.no_grad():
            # Get speech embeddings
            speech_embeddings = self.get_speech_embeddings(audio_features)
            projected_embeddings = self.adapter(speech_embeddings)
            
            # Get instruction embeddings
            instruction_ids = self.tokenizer(
                instruction.lower(),
                return_tensors="pt",
                padding=False
            ).input_ids.to(self.device)
            instruction_embeddings = self.language_model.get_input_embeddings()(instruction_ids)
            
            # Combine embeddings
            current_embeddings = torch.cat([
                instruction_embeddings.expand(projected_embeddings.size(0), -1, -1),
                projected_embeddings
            ], dim=1)
            
            transcriptions = []
            batch_size = audio_features.size(0)
            
            for b in range(batch_size):
                batch_embeddings = current_embeddings[b:b+1]
                generated_ids = []
                
                for _ in range(max_new_tokens):
                    outputs = self.language_model(inputs_embeds=batch_embeddings)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1)
                    
                    if next_token_id.item() == self.tokenizer.eos_token_id:
                        break
                    
                    generated_ids.append(next_token_id.item())
                    next_token_embedding = self.language_model.get_input_embeddings()(
                        next_token_id.unsqueeze(0)
                    )
                    batch_embeddings = torch.cat([batch_embeddings, next_token_embedding], dim=1)
                
                transcription = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                transcriptions.append(transcription.strip().lower())
            
            return transcriptions
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        adapter_params = sum(p.numel() for p in self.adapter.parameters())
        lm_params = sum(p.numel() for p in self.language_model.parameters())
        speech_params = sum(p.numel() for p in self.speech_encoder.parameters())
        
        return {
            "total": total,
            "trainable": trainable,
            "adapter": adapter_params,
            "language_model": lm_params,
            "speech_encoder": speech_params
        }


class EarlyStopping:
    """Early stopping based on validation metric"""
    
    def __init__(self, patience: int, metric: str = "cer", mode: str = "min"):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.mode == "min":
            improved = value < self.best_value
        else:
            improved = value > self.best_value
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """Training manager for ASR-Doge"""
    
    def __init__(
        self,
        model: ASRDogeModel,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        config: TrainingConfig
    ):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.config = config
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            [
                {"params": model.adapter.parameters(), "lr": config.learning_rate},
                {"params": model.language_model.parameters(), "lr": config.learning_rate}
            ],
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * config.epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            metric=config.early_stopping_metric
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.fp16 else None
        
        # Tracking
        self.global_step = 0
        self.best_metric = float("inf")
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # Move data to device
            audio = batch["audio"].to(self.config.device)
            
            # Tokenize text labels
            labels = self.model.tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).input_ids.to(self.config.device)
            
            # Process audio through speech encoder's processor
            audio_features = self.model.speech_processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.config.device)
            
            # Forward pass
            if self.config.fp16 and self.scaler is not None:
                with autocast():
                    outputs = self.model(audio_features, labels=labels)
                    loss = outputs["loss"] / self.config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(audio_features, labels=labels)
                loss = outputs["loss"] / self.config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Update weights
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Logging
            if self.global_step % self.config.log_steps == 0:
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        "train/loss": loss.item() * self.config.gradient_accumulation_steps,
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/step": self.global_step
                    })
            
            # Evaluation
            if self.global_step % self.config.eval_steps == 0:
                metrics = self.evaluate()
                
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        "eval/loss": metrics.get("loss", 0),
                        "eval/wer": metrics.get("wer", 0),
                        "eval/cer": metrics.get("cer", 0),
                        "eval/step": self.global_step
                    })
                
                # Early stopping check
                metric_value = metrics.get(self.config.early_stopping_metric, float("inf"))
                if self.early_stopping(metric_value):
                    logger.info(f"Early stopping triggered at step {self.global_step}")
                    return total_loss / num_batches
                
                # Save best model
                if metric_value < self.best_metric:
                    self.best_metric = metric_value
                    self.save_checkpoint("best")
                
                self.model.train()
            
            # Regular checkpointing
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(f"step_{self.global_step}")
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, max_samples: Optional[int] = None) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        
        max_samples = max_samples or self.config.max_eval_samples
        total_loss = 0
        all_predictions = []
        all_references = []
        num_samples = 0
        
        for batch in tqdm(self.dev_loader, desc="Evaluating"):
            if num_samples >= max_samples:
                break
            
            audio = batch["audio"].to(self.config.device)
            texts = batch["text"]
            
            # Process audio
            audio_features = self.model.speech_processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.config.device)
            
            # Compute loss with teacher forcing
            labels = self.model.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).input_ids.to(self.config.device)
            
            outputs = self.model(audio_features, labels=labels)
            total_loss += outputs["loss"].item()
            
            # Generate predictions
            predictions = self.model.generate(audio_features)
            
            all_predictions.extend(predictions)
            all_references.extend([t.lower() for t in texts])
            num_samples += len(texts)
        
        # Compute metrics
        metrics = {"loss": total_loss / (num_samples / self.config.batch_size)}
        
        if JIWER_AVAILABLE and all_predictions and all_references:
            # Clean texts for metric computation
            clean_preds = [p.lower().strip() for p in all_predictions]
            clean_refs = [r.lower().strip() for r in all_references]
            
            # Filter empty strings
            valid_pairs = [(p, r) for p, r in zip(clean_preds, clean_refs) if p and r]
            if valid_pairs:
                preds, refs = zip(*valid_pairs)
                metrics["wer"] = wer(list(refs), list(preds))
                metrics["cer"] = cer(list(refs), list(preds))
        
        logger.info(f"Evaluation - Loss: {metrics['loss']:.4f}, "
                   f"WER: {metrics.get('wer', 'N/A'):.4f}, "
                   f"CER: {metrics.get('cer', 'N/A'):.4f}")
        
        return metrics
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save adapter
        torch.save(
            self.model.adapter.state_dict(),
            os.path.join(checkpoint_dir, "adapter.pth")
        )
        
        # Save language model
        torch.save(
            self.model.language_model.state_dict(),
            os.path.join(checkpoint_dir, "lm.pth")
        )
        
        # Save config
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump({
                "model_config": asdict(self.model.config),
                "training_config": asdict(self.config),
                "global_step": self.global_step,
                "best_metric": self.best_metric
            }, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint"""
        # Load adapter
        adapter_path = os.path.join(checkpoint_dir, "adapter.pth")
        if os.path.exists(adapter_path):
            self.model.adapter.load_state_dict(torch.load(adapter_path))
        
        # Load language model
        lm_path = os.path.join(checkpoint_dir, "lm.pth")
        if os.path.exists(lm_path):
            self.model.language_model.load_state_dict(torch.load(lm_path))
        
        # Load config
        config_path = os.path.join(checkpoint_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                saved_config = json.load(f)
                self.global_step = saved_config.get("global_step", 0)
                self.best_metric = saved_config.get("best_metric", float("inf"))
        
        logger.info(f"Loaded checkpoint from {checkpoint_dir}")
    
    def train(self):
        """Full training loop"""
        logger.info("Starting training...")
        logger.info(f"Model parameters: {self.model.count_parameters()}")
        
        for epoch in range(self.config.epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            logger.info(f"{'='*50}")
            
            avg_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
            
            # Full evaluation at end of epoch
            metrics = self.evaluate()
            
            if self.early_stopping.should_stop:
                logger.info("Training stopped early")
                break
        
        # Save final checkpoint
        self.save_checkpoint("final")
        logger.info("Training complete!")
        
        return self.best_metric


def main():
    parser = argparse.ArgumentParser(description="Train ASR-Doge model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing data manifests")
    parser.add_argument("--train_manifest", type=str, default="train_manifest.json")
    parser.add_argument("--dev_manifest", type=str, default="dev_manifest.json")
    
    # Model arguments
    parser.add_argument("--speech_encoder", type=str,
                        default="ibm-granite/granite-speech-3.3-2b")
    parser.add_argument("--language_model", type=str,
                        default="SmallDoge/Doge-320M-Checkpoint")
    parser.add_argument("--adapter_hidden_dim", type=int, default=512)
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Logging
    parser.add_argument("--wandb_project", type=str, default="asr-doge")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    
    # Resume
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Checkpoint directory to resume from")
    
    args = parser.parse_args()
    
    # Create configs
    model_config = ModelConfig(
        speech_encoder=args.speech_encoder,
        language_model=args.language_model,
        adapter_hidden_dim=args.adapter_hidden_dim
    )
    
    training_config = TrainingConfig(
        data_dir=args.data_dir,
        train_manifest=args.train_manifest,
        dev_manifest=args.dev_manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        device=args.device,
        fp16=args.fp16,
        num_workers=args.num_workers,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name or f"asr-doge-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Initialize wandb
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(
            project=training_config.wandb_project,
            entity=training_config.wandb_entity,
            name=training_config.wandb_run_name,
            config={
                "model": asdict(model_config),
                "training": asdict(training_config)
            }
        )
    
    # Create model
    logger.info("Creating model...")
    model = ASRDogeModel(model_config, device=args.device)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    from data.data_processor import (
        ASRDogeDataset, AudioProcessor, AudioConfig, collate_fn
    )
    
    audio_processor = AudioProcessor(AudioConfig())
    
    train_dataset = ASRDogeDataset(
        os.path.join(args.data_dir, args.train_manifest),
        audio_processor
    )
    dev_dataset = ASRDogeDataset(
        os.path.join(args.data_dir, args.dev_manifest),
        audio_processor
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create trainer
    trainer = Trainer(model, train_loader, dev_loader, training_config)
    
    # Resume from checkpoint
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Train
    best_metric = trainer.train()
    
    # Finish wandb
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.summary["best_metric"] = best_metric
        wandb.finish()
    
    logger.info(f"Best {training_config.early_stopping_metric}: {best_metric:.4f}")


if __name__ == "__main__":
    main()

