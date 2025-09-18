#!/usr/bin/env python3
"""
SVG Generator - Inference Script
Lädt ein trainiertes Model und generiert SVGs aus Text-Prompts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional
import json
import math
import random
import string


class TrainingConfig:
    """Konfiguration für das Training - muss für Model-Loading vorhanden sein"""
    
    def __init__(self):
        # Daten
        self.data_dir = "./dataset"
        self.output_dir = "./outputs"
        self.val_split = 0.1
        
        # Modell
        self.d_model = 512  # Angepasst an neue Architektur
        self.num_heads = 8
        self.num_layers = 6
        self.dropout = 0.1
        self.max_shapes = 8
        
        # Text
        self.max_text_length = 64
        
        # Training
        self.batch_size = 16
        self.learning_rate = 2e-4
        self.weight_decay = 0.01
        self.grad_clip = 1.0
        
        # Logging
        self.log_interval = 50
        self.eval_interval = 200
        
        # System
        self.num_workers = 2


class SVGGenerator:
    def __init__(self, model_path: str = "./outputs/best_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Lade Model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.text_tokenizer = checkpoint['text_tokenizer']
        config = checkpoint['config']
        
        # Erstelle Model-Instanz mit neuer Architektur
        self.model = SVGParameterModel(
            text_vocab_size=len(self.text_tokenizer.get_vocab()),
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_text_length=config.max_text_length,
            dropout=config.dropout,
            max_shapes=config.max_shapes
        ).to(self.device)
        
        # Gib dem Model Zugriff auf den Tokenizer
        self.model.text_tokenizer = self.text_tokenizer
        
        # Lade Model State
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model geladen mit validation loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    def generate(self, prompt: str) -> str:
        """Generiert SVG aus Text-Prompt"""
        # Tokenisiere Input
        text_tokens = self.text_tokenizer.encode(
            prompt,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Erstelle Mask
        text_mask = (text_tokens == self.text_tokenizer.pad_token_id)
        
        # Generiere SVG
        with torch.no_grad():
            svg_strings = self.model.generate_svg(text_tokens, text_mask)
            return svg_strings[0]
    
    def save_svg(self, svg_content: str, filename: str):
        """Speichert SVG in Datei"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        print(f"SVG gespeichert: {filename}")


# Aktualisierte SVGParameterModel Klasse
class SVGParameterModel(nn.Module):
    """
    Model das komplexe SVG-Strukturen wie Smileys generiert
    """
    
    def __init__(self, 
                 text_vocab_size: int,
                 d_model: int = 512,  # Größer für komplexe Formen
                 num_heads: int = 8,   
                 num_layers: int = 6,  
                 max_text_length: int = 64,
                 dropout: float = 0.1,
                 max_shapes: int = 8):
        super().__init__()
        
        self.d_model = d_model
        self.max_text_length = max_text_length
        self.max_shapes = max_shapes
        
        # Text Encoder
        self.text_embedding = nn.Embedding(text_vocab_size, d_model)
        self.text_pos_encoding = self._create_positional_encoding(max_text_length, d_model)
        
        # Transformer Encoder
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, d_model*4, dropout, batch_first=True),
            num_layers
        )
        
        # SVG Code Generator - generiert direkt SVG-ähnliche Tokens
        self.svg_generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1024)  # Große Output-Dimension für komplexe SVGs
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_parameters()
    
    def _create_positional_encoding(self, max_length: int, d_model: int):
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, text_tokens, text_mask=None):
        batch_size, seq_len = text_tokens.size()
        
        # Text Embeddings
        text_embed = self.text_embedding(text_tokens) * np.sqrt(self.d_model)
        text_embed = text_embed + self.text_pos_encoding[:, :seq_len, :]
        text_embed = self.dropout(text_embed)
        
        # Text Encoding
        if text_mask is not None:
            text_mask = ~text_mask.squeeze(1).squeeze(1).bool()
        
        encoded = self.text_encoder(text_embed, src_key_padding_mask=text_mask)
        
        # Pooling
        if text_mask is not None:
            mask_expanded = text_mask.unsqueeze(-1).expand_as(encoded)
            encoded = encoded.masked_fill(mask_expanded, 0)
            lengths = (~text_mask).sum(dim=1, keepdim=True).float()
            pooled = encoded.sum(dim=1) / lengths
        else:
            pooled = encoded.mean(dim=1)
        
        # Generiere SVG-Features
        svg_features = self.svg_generator(pooled)
        
        return {'svg_features': svg_features}

    def generate_svg(self, text_tokens, text_mask=None):
        """Generiert SVG direkt aus Features"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(text_tokens, text_mask)
            svg_features = outputs['svg_features'][0]  # Erstes Batch-Element
            
            return [self._features_to_svg(svg_features, text_tokens[0])]
    
    def _features_to_svg(self, features, text_tokens):
        """Konvertiert Features zu SVG - ECHTE KI-basierte Generierung!"""
        svg_parts = ['<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">']
        
        # Nutze die ECHTEN KI-Features!
        features_np = features.cpu().numpy()
        
        # Debug: zeige was die KI wirklich generiert hat
        print(f"DEBUG: KI Features (erste 20): {features_np[:20]}")
        
        # Interpretiere Features als verschiedene Shapes
        # Jeder Feature-Block repräsentiert eine potentielle Shape
        
        num_shapes = min(8, max(1, int((features_np[0] * 8))))  # 1-8 Shapes
        print(f"DEBUG: KI will {num_shapes} Shapes generieren")
        
        for i in range(num_shapes):
            # Für jede Shape nehme verschiedene Feature-Bereiche
            shape_offset = i * 128  # Jede Shape bekommt 128 Features
            
            if shape_offset + 10 >= len(features_np):
                break
                
            # Shape-Typ basierend auf Features
            shape_type_val = features_np[shape_offset]
            
            # Position und Größe basierend auf KI-Features
            x = 10 + (features_np[shape_offset + 1] * 80)  # 10-90
            y = 10 + (features_np[shape_offset + 2] * 80)  # 10-90
            size = 5 + (features_np[shape_offset + 3] * 30)  # 5-35
            
            # Weitere Parameter
            param1 = features_np[shape_offset + 4] * 100
            param2 = features_np[shape_offset + 5] * 100
            param3 = features_np[shape_offset + 6] * 100
            
            if shape_type_val < 0.2:
                # KREIS
                svg_parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}" fill="black"/>')
                
            elif shape_type_val < 0.4:
                # RECHTECK
                width = 5 + (param1 % 40)
                height = 5 + (param2 % 40)
                svg_parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="black"/>')
                
            elif shape_type_val < 0.6:
                # LINIE
                x2 = 10 + (param1 % 80)
                y2 = 10 + (param2 % 80)
                svg_parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="black" stroke-width="2"/>')
                
            elif shape_type_val < 0.8:
                # ELLIPSE
                rx = 5 + (param1 % 25)
                ry = 5 + (param2 % 25)
                svg_parts.append(f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="black"/>')
                
            else:
                # PFAD (einfach)
                x2 = x + (-20 + (param1 % 40))
                y2 = y + (-20 + (param2 % 40))
                x3 = x + (-20 + (param3 % 40))
                y3 = y + (-10 + ((param1 + param2) % 20))
                svg_parts.append(f'<path d="M {x:.1f},{y:.1f} L {x2:.1f},{y2:.1f} L {x3:.1f},{y3:.1f}" stroke="black" stroke-width="2" fill="none"/>')
        
        svg_parts.append('</svg>')
        result = '\n'.join(svg_parts)
        
        print(f"DEBUG: Generiertes SVG hat {len(svg_parts)-2} Elemente")
        return result


def main():
    """Hauptfunktion für interaktive SVG-Generierung"""
    print("🎨 SVG Generator - Text zu SVG Konverter")
    print("=========================================")
    
    try:
        # Lade Generator
        generator = SVGGenerator()
        
        print("\n✅ Model geladen! Bereit für Prompts.")
        print("💡 Beispiele: 'happy smiley', 'sad face', 'circle', 'square'")
        print("❌ Zum Beenden 'quit' eingeben")
        
        counter = 1
        while True:
            print(f"\n📝 Prompt #{counter}:")
            prompt = input("> ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Auf Wiedersehen!")
                break
            
            if not prompt:
                print("⚠️  Leerer Prompt! Bitte Text eingeben.")
                continue
            
            try:
                # Generiere SVG
                print("🔄 Generiere SVG...")
                svg_content = generator.generate(prompt)
                
                # Zeige SVG
                print(f"\n🎨 Generiertes SVG für '{prompt}':")
                print("=" * 50)
                print(svg_content)
                print("=" * 50)
                
                # Speichere SVG in outputs Ordner
                import os
                os.makedirs("outputs", exist_ok=True)
                filename = f"outputs/generated_svg_{counter:03d}.svg"
                generator.save_svg(svg_content, filename)
                
                print(f"💾 SVG gespeichert als: {filename}")
                counter += 1
                
            except Exception as e:
                print(f"❌ Fehler bei Generierung: {e}")
    
    except Exception as e:
        print(f"❌ Fehler beim Laden des Models: {e}")


if __name__ == "__main__":
    main()
