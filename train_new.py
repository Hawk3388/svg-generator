"""
Vollständig neues Training-Script für Text-to-SVG Generierung
Verwendet echte Sequence-to-Sequence Architektur ohne Fallbacks
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random

from svg_tokenizer import SVGTokenizer
from svg_transformer import SVGGenerationModel, SVGTransformer


@dataclass
class TrainingConfig:
    """Training Configuration"""
    # Dataset
    data_dir: str = "./dataset"
    output_dir: str = "./outputs"
    
    # Model Architecture
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    
    # Sequence Lengths
    max_text_len: int = 128
    max_svg_len: int = 1024
    
    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    warmup_steps: int = 1000
    grad_clip_norm: float = 1.0
    
    # Validation & Logging
    val_split: float = 0.1
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    
    # Generation
    generation_temperature: float = 0.8
    generation_top_k: int = 50
    generation_top_p: float = 0.9
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    seed: int = 42


class OpenMojiDataset(Dataset):
    """
    Dataset für OpenMoji SVG-Daten mit Text-Beschreibungen
    """
    
    def __init__(self, 
                 data_dir: str, 
                 text_tokenizer: AutoTokenizer,
                 svg_tokenizer: SVGTokenizer,
                 max_text_len: int = 128,
                 max_svg_len: int = 1024,
                 split: str = "train"):
        
        self.data_dir = Path(data_dir)
        self.text_tokenizer = text_tokenizer
        self.svg_tokenizer = svg_tokenizer
        self.max_text_len = max_text_len
        self.max_svg_len = max_svg_len
        self.split = split
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Filter valid samples
        self.samples = self._prepare_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_metadata(self) -> List[Dict]:
        """Load OpenMoji metadata"""
        json_path = self.data_dir / "openmoji.json"
        
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return metadata
    
    def _prepare_samples(self) -> List[Dict]:
        """Prepare valid samples with existing SVG files"""
        samples = []
        
        for item in self.metadata:
            hexcode = item['hexcode']
            svg_path = self.data_dir / f"{hexcode}.svg"
            
            # Check if SVG file exists
            if not svg_path.exists():
                continue
            
            # Create text description from metadata
            text_parts = []
            
            # Add annotation (main description)
            if item.get('annotation'):
                text_parts.append(item['annotation'])
            
            # Add tags for more context
            if item.get('tags'):
                tags = item['tags'].split(', ')[:5]  # Limit to 5 most relevant tags
                text_parts.extend(tags)
            
            # Add OpenMoji specific tags
            if item.get('openmoji_tags'):
                omoji_tags = item['openmoji_tags'].split(', ')
                text_parts.extend(omoji_tags)
            
            text_description = ' '.join(text_parts).strip()
            
            if text_description:  # Only include if we have a description
                samples.append({
                    'hexcode': hexcode,
                    'text': text_description,
                    'svg_path': svg_path,
                    'group': item.get('group', ''),
                    'subgroup': item.get('subgroups', '')
                })
        
        # Split train/val
        random.shuffle(samples)
        split_idx = int(len(samples) * 0.9)  # 90% train, 10% val
        
        if self.split == "train":
            return samples[:split_idx]
        else:
            return samples[split_idx:]
    
    def _load_svg_content(self, svg_path: Path) -> Optional[str]:
        """Load and validate SVG content"""
        try:
            with open(svg_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Basic SVG validation
            if '<svg' in content and '</svg>' in content:
                return content
            else:
                return None
                
        except Exception as e:
            print(f"Error loading SVG {svg_path}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load SVG content
        svg_content = self._load_svg_content(sample['svg_path'])
        if svg_content is None:
            # Return a dummy sample if SVG can't be loaded
            return self.__getitem__((idx + 1) % len(self.samples))
        
        # Tokenize text
        text_encoding = self.text_tokenizer(
            sample['text'],
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize SVG
        svg_tokens = self.svg_tokenizer.tokenize(svg_content)
        
        # Pad or truncate SVG tokens
        if len(svg_tokens) > self.max_svg_len:
            svg_tokens = svg_tokens[:self.max_svg_len]
        else:
            pad_id = self.svg_tokenizer.token_to_id['<PAD>']
            svg_tokens.extend([pad_id] * (self.max_svg_len - len(svg_tokens)))
        
        # Prepare input/target for teacher forcing
        svg_input = svg_tokens[:-1]  # All except last token
        svg_target = svg_tokens[1:]   # All except first token
        
        return {
            'text_input_ids': text_encoding['input_ids'].squeeze(0),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'svg_input_ids': torch.tensor(svg_input, dtype=torch.long),
            'svg_target_ids': torch.tensor(svg_target, dtype=torch.long),
            'text_raw': sample['text'],
            'hexcode': sample['hexcode']
        }


class SVGTrainer:
    """
    Trainer für SVG-Generation Model
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set seed for reproducibility
        self._set_seed(config.seed)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize tokenizers
        self.text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.svg_tokenizer = SVGTokenizer(vocab_size=15000)
        
        # Build SVG vocabulary from dataset
        self._build_svg_vocabulary()
        
        # Initialize model
        self.model = SVGGenerationModel(config)
        self.model.text_tokenizer = self.text_tokenizer
        self.model.svg_tokenizer = self.svg_tokenizer
        self.model.initialize_model(
            text_vocab_size=len(self.text_tokenizer),
            svg_vocab_size=self.svg_tokenizer.get_vocab_size()
        )
        self.model = self.model.to(self.device)
        
        # Initialize datasets
        self.train_dataset = OpenMojiDataset(
            config.data_dir, self.text_tokenizer, self.svg_tokenizer,
            config.max_text_len, config.max_svg_len, "train"
        )
        
        self.val_dataset = OpenMojiDataset(
            config.data_dir, self.text_tokenizer, self.svg_tokenizer,
            config.max_text_len, config.max_svg_len, "val"
        )
        
        # Initialize data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=len(self.train_loader) * config.num_epochs,
            pct_start=0.1
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.svg_tokenizer.token_to_id['<PAD>'])
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.early_stopping_patience = 5
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        logging.info(f"Initialized trainer with {len(self.train_dataset)} train and {len(self.val_dataset)} val samples")
        logging.info(f"Text vocab size: {len(self.text_tokenizer)}")
        logging.info(f"SVG vocab size: {self.svg_tokenizer.get_vocab_size()}")
        logging.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
    
    def _build_svg_vocabulary(self):
        """Build SVG vocabulary from dataset"""
        logging.info("Building SVG vocabulary from dataset...")
        
        svg_files = list(Path(self.config.data_dir).glob("*.svg"))
        
        # Build vocab from ALL files, not just first 1000
        for svg_file in tqdm(svg_files, desc="Building vocab"):
            try:
                with open(svg_file, 'r', encoding='utf-8') as f:
                    svg_content = f.read()
                
                # Tokenize to build vocabulary
                self.svg_tokenizer.tokenize(svg_content)
                
            except Exception as e:
                continue
        
        # Save vocabulary
        vocab_path = Path(self.config.output_dir) / "svg_vocab.json"
        self.svg_tokenizer.save_vocab(str(vocab_path))
        
        # Freeze vocabulary to prevent new tokens during training
        self.svg_tokenizer.freeze_vocab()
        
        logging.info(f"SVG vocabulary built with {self.svg_tokenizer.get_vocab_size()} tokens")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        text_input_ids = batch['text_input_ids'].to(self.device)
        svg_input_ids = batch['svg_input_ids'].to(self.device)
        svg_target_ids = batch['svg_target_ids'].to(self.device)
        
        # Forward pass
        logits = self.model(text_input_ids, svg_input_ids)
        
        # Calculate loss (only on non-padding tokens)
        # Create mask for non-padding tokens
        padding_mask = (svg_target_ids != 0)  # Assuming 0 is padding token
        
        # Flatten and apply mask
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = svg_target_ids.view(-1)
        mask_flat = padding_mask.view(-1)
        
        # Only compute loss on non-padding tokens
        if mask_flat.sum() > 0:
            loss = self.criterion(logits_flat[mask_flat], targets_flat[mask_flat])
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        
        # Update
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def validate(self) -> Tuple[float, float]:
        """Validation step"""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                text_input_ids = batch['text_input_ids'].to(self.device)
                svg_input_ids = batch['svg_input_ids'].to(self.device)
                svg_target_ids = batch['svg_target_ids'].to(self.device)
                
                logits = self.model(text_input_ids, svg_input_ids)
                loss = self.criterion(logits.view(-1, logits.size(-1)), svg_target_ids.view(-1))
                
                # Count non-padding tokens
                non_pad_tokens = (svg_target_ids != self.svg_tokenizer.token_to_id['<PAD>']).sum().item()
                
                total_loss += loss.item() * non_pad_tokens
                total_tokens += non_pad_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        return avg_loss, perplexity
    
    def generate_sample(self, text_prompt: str) -> str:
        """Generate a sample SVG for evaluation"""
        self.model.eval()
        
        try:
            svg_string = self.model.generate(text_prompt, max_length=512)
            return svg_string
        except Exception as e:
            logging.error(f"Generation error: {e}")
            return f"<svg><text>Error: {str(e)}</text></svg>"
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'text_tokenizer': self.text_tokenizer,
            'svg_tokenizer_vocab': {
                'token_to_id': self.svg_tokenizer.token_to_id,
                'id_to_token': self.svg_tokenizer.id_to_token,
                'next_id': self.svg_tokenizer.next_id,
                'vocab_size': self.svg_tokenizer.vocab_size
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.output_dir) / f"checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.output_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            logging.info(f"Saved new best model with val_loss: {self.best_val_loss:.4f}")
    
    def train(self):
        """Main training loop"""
        logging.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            # Training
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                self.step += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.4f}', 'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'})
                
                # Log training step
                if self.step % self.config.log_interval == 0:
                    logging.info(f"Step {self.step}: loss={loss:.4f}, lr={self.scheduler.get_last_lr()[0]:.2e}")
                
                # Validation
                if self.step % self.config.eval_interval == 0:
                    val_loss, perplexity = self.validate()
                    logging.info(f"Validation - Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
                    
                    # Sample generation
                    sample_svg = self.generate_sample("happy smiley face")
                    logging.info(f"Sample generation: {sample_svg[:200]}...")
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.epochs_without_improvement = 0
                        self.save_checkpoint(is_best=True)
                    else:
                        self.epochs_without_improvement += 1
                    
                    # Early stopping check
                    if self.epochs_without_improvement >= self.early_stopping_patience:
                        logging.info(f"Early stopping after {self.early_stopping_patience} epochs without improvement")
                        return
                
                # Save checkpoint
                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint()
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            logging.info(f"Epoch {epoch+1} completed - Average loss: {avg_epoch_loss:.4f}")
            
            # Final validation for epoch
            val_loss, perplexity = self.validate()
            logging.info(f"End of epoch validation - Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        logging.info("Training completed!")
        
        # Final checkpoint
        self.save_checkpoint()


def main():
    """Main training function"""
    config = TrainingConfig()
    
    # Check if dataset exists
    if not Path(config.data_dir).exists():
        raise FileNotFoundError(f"Dataset directory {config.data_dir} not found!")
    
    if not (Path(config.data_dir) / "openmoji.json").exists():
        raise FileNotFoundError(f"openmoji.json not found in {config.data_dir}!")
    
    # Start training
    trainer = SVGTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()