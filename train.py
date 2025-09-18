"""
PyTorch Trainingsskript für Text-to-SVG Generierung mit OpenMoji Dataset
Generiert SVG-Dateien aus Text-Prompts mithilfe eines Transformer-basierten Modells
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import xml.etree.ElementTree as ET
from transformers import AutoTokenizer
import logging
import random

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SVGTokenizer:
    """
    Spezieller Tokenizer für SVG-Inhalte
    Behandelt SVG-Tags, Attribute und Pfade als spezielle Tokens
    """
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<SOS>': 1,  # Start of Sequence
            '<EOS>': 2,  # End of Sequence
            '<UNK>': 3,  # Unknown
        }
        
        # Grundlegende SVG-Tokens
        self.svg_tokens = [
            'svg', 'path', 'circle', 'rect', 'ellipse', 'line', 'polygon', 'polyline',
            'g', 'defs', 'use', 'text', 'tspan', 'image', 'clipPath', 'mask',
            'd=', 'cx=', 'cy=', 'r=', 'rx=', 'ry=', 'x=', 'y=', 'width=', 'height=',
            'fill=', 'stroke=', 'stroke-width=', 'transform=', 'viewBox=',
            'M', 'L', 'H', 'V', 'C', 'S', 'Q', 'T', 'A', 'Z', 'm', 'l', 'h', 'v', 'c', 's', 'q', 't', 'a', 'z',
            '"', "'", '=', '<', '>', '/', ' ', '\n', '\t'
        ]
        
        self._build_vocab()
    
    def _build_vocab(self):
        """Erstellt das Vokabular"""
        current_id = len(self.special_tokens)
        
        # Special tokens hinzufügen
        for token, token_id in self.special_tokens.items():
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        
        # SVG tokens hinzufügen
        for token in self.svg_tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
        
        # Zahlen und Buchstaben hinzufügen
        for i in range(10):
            token = str(i)
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
        
        for i in range(26):
            token = chr(ord('a') + i)
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
            
            token = chr(ord('A') + i)
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
    
    def tokenize(self, svg_content: str) -> List[str]:
        """Tokenisiert SVG-Inhalt"""
        # Einfache Tokenisierung basierend auf Zeichen und bekannten SVG-Elementen
        tokens = []
        i = 0
        while i < len(svg_content):
            # Versuche längste Übereinstimmung zu finden
            found = False
            for length in range(min(20, len(svg_content) - i), 0, -1):
                substring = svg_content[i:i+length]
                if substring in self.token_to_id:
                    tokens.append(substring)
                    i += length
                    found = True
                    break
            
            if not found:
                # Einzelnes Zeichen hinzufügen oder als UNK markieren
                char = svg_content[i]
                if char in self.token_to_id:
                    tokens.append(char)
                else:
                    tokens.append('<UNK>')
                i += 1
        
        return tokens
    
    def encode(self, svg_content: str, max_length: int = 512) -> List[int]:
        """Kodiert SVG-Inhalt zu Token-IDs"""
        tokens = ['<SOS>'] + self.tokenize(svg_content) + ['<EOS>']
        
        # Auf maximale Länge kürzen oder auffüllen
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + ['<EOS>']
        else:
            tokens = tokens + ['<PAD>'] * (max_length - len(tokens))
        
        return [self.token_to_id.get(token, self.token_to_id['<UNK>']) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Dekodiert Token-IDs zurück zu SVG-Inhalt"""
        tokens = [self.id_to_token.get(token_id, '<UNK>') for token_id in token_ids]
        
        # Entferne special tokens
        decoded_tokens = []
        for token in tokens:
            if token in ['<SOS>', '<EOS>', '<PAD>']:
                if token == '<EOS>':
                    break
                continue
            decoded_tokens.append(token)
        
        return ''.join(decoded_tokens)


class OpenMojiDataset(Dataset):
    """
    Dataset-Klasse für OpenMoji-Daten
    Lädt Emoji-SVGs und ihre Beschreibungen
    """
    
    def __init__(self, data_dir: str, text_tokenizer, 
                 max_text_length: int = 64):
        self.data_dir = Path(data_dir)
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        
        # Lade OpenMoji Metadaten
        self.metadata = self._load_metadata()
        self.samples = self._prepare_samples()
        
        logger.info(f"Dataset geladen: {len(self.samples)} Samples")
    
    def _load_metadata(self) -> List[Dict]:
        """Lädt OpenMoji Metadaten"""
        metadata_file = self.data_dir / "openmoji.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadaten-Datei nicht gefunden: {metadata_file}")
        
        logger.info(f"Lade Metadaten aus: {metadata_file}")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    
    def _prepare_samples(self) -> List[Dict]:
        """Bereitet die Trainingssamples vor"""
        samples = []
        
        for item in self.metadata:
            hexcode = item['hexcode']
            svg_path = self.data_dir / f"{hexcode}.svg"
            
            if svg_path.exists():
                # Text-Beschreibung erstellen
                text_parts = []
                if 'annotation' in item and item['annotation']:
                    text_parts.append(item['annotation'])
                if 'tags' in item and item['tags']:
                    # Tags sind bereits komma-separiert
                    text_parts.extend([tag.strip() for tag in item['tags'].split(',')])
                if 'openmoji_tags' in item and item['openmoji_tags']:
                    text_parts.extend([tag.strip() for tag in item['openmoji_tags'].split(',')])
                
                text_description = " ".join(text_parts).strip()
                if text_description:
                    samples.append({
                        'hexcode': hexcode,
                        'svg_path': svg_path,
                        'text': text_description,
                        'metadata': item
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Lade SVG-Inhalt
        with open(sample['svg_path'], 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        # Bereinige SVG-Inhalt und extrahiere Parameter
        svg_content = self._clean_svg(svg_content)
        svg_params = self._extract_svg_parameters(svg_content)
        
        # Tokenisiere Text
        text_tokens = self.text_tokenizer.encode(
            sample['text'], 
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).squeeze()
        
        return {
            'text_tokens': text_tokens,
            'svg_params': svg_params,
            'text': sample['text'],
            'svg_content': svg_content,
            'hexcode': sample['hexcode']
        }
    
    def _clean_svg(self, svg_content: str) -> str:
        """Bereinigt SVG-Inhalt"""
        # Entferne Kommentare
        svg_content = re.sub(r'<!--.*?-->', '', svg_content, flags=re.DOTALL)
        
        # Vereinfache Whitespace
        svg_content = re.sub(r'\s+', ' ', svg_content)
        svg_content = svg_content.strip()
        
        return svg_content
    
    def _extract_svg_parameters(self, svg_content: str) -> Dict:
        """Extrahiert ALLE SVG-Elemente für komplexe Formen wie Smileys"""
        params = {
            'svg_code': svg_content,  # Speichere den ganzen SVG-Code
            'elements': []
        }
        
        try:
            root = ET.fromstring(svg_content)
            
            # Extrahiere ALLE Elemente
            for elem in root.iter():
                if elem.tag.endswith(('circle', 'ellipse', 'rect', 'path', 'polygon', 'line')):
                    element = {
                        'type': elem.tag.split('}')[-1],  # Entferne namespace
                        'attributes': dict(elem.attrib)
                    }
                    params['elements'].append(element)
        
        except ET.ParseError:
            pass
        
        return self._encode_to_sequence(params)
    
    def _encode_to_sequence(self, params: Dict) -> torch.Tensor:
        """Kodiert SVG als großen Feature-Vektor für komplexe Formen"""
        svg_code = params.get('svg_code', '')
        
        # Erstelle 1024-dimensionalen Feature-Vektor
        features = torch.zeros(1024)
        
        # Analysiere SVG-Text für verschiedene Muster
        svg_lower = svg_code.lower()
        
        # Pattern-Detection für Smileys
        if 'smiley' in svg_lower or 'smile' in svg_lower:
            features[0] = 1.0  # Smiley-Flag
        
        # Extrahiere alle Zahlen aus dem SVG
        numbers = re.findall(r'[-+]?\d*\.?\d+', svg_code)
        
        # Setze Koordinaten in Features (normalisiert)
        for i, num_str in enumerate(numbers[:500]):  # Bis zu 500 Zahlen
            try:
                features[i + 50] = float(num_str) / 100.0  # Normalisiere
            except:
                features[i + 50] = 0.0
        
        # Shape-Type Detection
        if 'circle' in svg_lower:
            features[1] = 1.0
        if 'rect' in svg_lower:
            features[2] = 1.0  
        if 'path' in svg_lower:
            features[3] = 1.0
        if 'polygon' in svg_lower:
            features[4] = 1.0
            
        # Farb-Detection (nur schwarz/weiß)
        if 'fill="black"' in svg_code or 'fill="#000"' in svg_code:
            features[5] = 0.0  # schwarz
        else:
            features[5] = 1.0  # weiß
            
        # Komplexitäts-Features
        elements = params.get('elements', [])
        features[6] = min(1.0, len(elements) / 10.0)
        
        # Text-Pattern Features
        patterns = ['happy', 'sad', 'face', 'eye', 'mouth', 'nose', 'head']
        for i, pattern in enumerate(patterns):
            if pattern in svg_lower:
                features[10 + i] = 1.0
        
        return features
    
    def _parse_color(self, color_str: str) -> Tuple[int, int, int]:
        """Parsed Farb-String zu RGB"""
        if color_str.startswith('#'):
            hex_color = color_str[1:]
            if len(hex_color) == 6:
                return (
                    int(hex_color[0:2], 16),
                    int(hex_color[2:4], 16),
                    int(hex_color[4:6], 16)
                )
        elif color_str.startswith('rgb'):
            # rgb(r,g,b) parsing
            nums = re.findall(r'\d+', color_str)
            if len(nums) >= 3:
                return (int(nums[0]), int(nums[1]), int(nums[2]))
        
        # Default schwarz
        return (0, 0, 0)
    
    def _parse_simple_path(self, d: str) -> Optional[List[float]]:
        """Parsed einfache Pfade (M x,y L x,y)"""
        coords = re.findall(r'[-+]?\d*\.?\d+', d)
        if len(coords) >= 4:
            return [float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])]
        return None
    
    def _normalize_parameters(self, params: Dict) -> torch.Tensor:
        """Erstellt Ziel-Parameter basierend auf Text-Beschreibung"""
        max_shapes = 8
        shape_data = torch.zeros(13, max_shapes)
        
        # Vereinfacht: Erstelle Standard-Shape basierend auf erstem gefundenem Shape
        if params['circles']:
            # Circle als primäre Form
            circle = params['circles'][0]
            shape_data[0, 0] = 1.0  # circle_present
            shape_data[1, 0] = min(1.0, circle['cx'] / 100.0)  # cx
            shape_data[2, 0] = min(1.0, circle['cy'] / 100.0)  # cy
            shape_data[3, 0] = min(1.0, circle['r'] / 50.0)    # r
            shape_data[4, 0] = circle['fill'][0] / 255.0  # R
            shape_data[5, 0] = circle['fill'][1] / 255.0  # G 
            shape_data[6, 0] = circle['fill'][2] / 255.0  # B
            
        elif params['rects']:
            # Rectangle als primäre Form
            rect = params['rects'][0]
            shape_data[7, 0] = 1.0  # rect_present
            shape_data[8, 0] = min(1.0, rect['x'] / 100.0)      # x
            shape_data[9, 0] = min(1.0, rect['y'] / 100.0)      # y
            shape_data[10, 0] = min(1.0, rect['width'] / 50.0)  # w
            shape_data[11, 0] = min(1.0, rect['height'] / 50.0) # h
            shape_data[4, 0] = rect['fill'][0] / 255.0  # R
            shape_data[5, 0] = rect['fill'][1] / 255.0  # G 
            shape_data[6, 0] = rect['fill'][2] / 255.0  # B
            
        elif params['paths']:
            # Path als primäre Form
            path = params['paths'][0]
            shape_data[12, 0] = 1.0  # path_present
            shape_data[4, 0] = path['stroke'][0] / 255.0  # R
            shape_data[5, 0] = path['stroke'][1] / 255.0  # G 
            shape_data[6, 0] = path['stroke'][2] / 255.0  # B
            
        else:
            # Fallback: Default circle
            shape_data[0, 0] = 1.0  # circle_present
            shape_data[1, 0] = 0.5  # cx center
            shape_data[2, 0] = 0.5  # cy center  
            shape_data[3, 0] = 0.4  # r medium
            shape_data[4, 0] = 0.5  # R
            shape_data[5, 0] = 0.5  # G 
            shape_data[6, 0] = 0.5  # B
        
        return shape_data


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Mechanismus"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.W_o(attention_output)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """Transformer Block mit Self-Attention und Feed-Forward"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x, self_attention_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention (mit Encoder-Output)
        if encoder_output is not None:
            cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, cross_attention_mask)
            x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


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
            
            return [self._features_to_svg(svg_features)]
    
    def _features_to_svg(self, features):
        """Konvertiert Features zu SVG - nur schwarz/weiß"""
        svg_parts = ['<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">']
        
        # Interpretiere Features als Koordinaten und Formen
        features = torch.sigmoid(features) * 100  # 0-100 Bereich
        
        # Generiere verschiedene Formen basierend auf Feature-Mustern
        feature_vals = features.cpu().numpy()
        
        # Einfache Heuristik für verschiedene Formen
        if feature_vals[0] > 50:  # Smiley-Pattern
            # Gesicht
            svg_parts.append('<circle cx="50" cy="50" r="40" fill="none" stroke="black" stroke-width="2"/>')
            # Augen
            left_eye_x = 35 + (feature_vals[1] % 10)
            right_eye_x = 65 + (feature_vals[2] % 10)
            eye_y = 40 + (feature_vals[3] % 10)
            svg_parts.append(f'<circle cx="{left_eye_x:.1f}" cy="{eye_y:.1f}" r="3" fill="black"/>')
            svg_parts.append(f'<circle cx="{right_eye_x:.1f}" cy="{eye_y:.1f}" r="3" fill="black"/>')
            # Mund
            mouth_y = 60 + (feature_vals[4] % 10)
            svg_parts.append(f'<path d="M 35,{mouth_y:.1f} Q 50,{mouth_y+10:.1f} 65,{mouth_y:.1f}" stroke="black" stroke-width="2" fill="none"/>')
            
        elif feature_vals[5] > 50:  # Geometrische Form
            x = 20 + (feature_vals[6] % 60)
            y = 20 + (feature_vals[7] % 60)
            size = 10 + (feature_vals[8] % 30)
            svg_parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{size:.1f}" height="{size:.1f}" fill="black"/>')
            
        else:  # Kreis
            cx = 30 + (feature_vals[9] % 40)
            cy = 30 + (feature_vals[10] % 40)
            r = 5 + (feature_vals[11] % 20)
            svg_parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="black"/>')
        
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)


def create_masks(text_tokens, svg_tokens, pad_token_id=0):
    """Erstellt Attention-Masken"""
    # Text mask (padding mask)
    text_mask = (text_tokens != pad_token_id).unsqueeze(1).unsqueeze(2)
    
    # SVG mask (causal mask + padding mask)
    batch_size, svg_seq_len = svg_tokens.size()
    
    # Padding mask
    svg_padding_mask = (svg_tokens != pad_token_id).unsqueeze(1).unsqueeze(2)
    
    # Causal mask
    causal_mask = torch.tril(torch.ones(svg_seq_len, svg_seq_len)).unsqueeze(0).unsqueeze(0)
    
    # Kombiniere masks
    svg_mask = svg_padding_mask & causal_mask.to(svg_tokens.device)
    
    return text_mask, svg_mask


def calculate_loss(model_outputs, targets, alpha=0.1):
    """
    Berechnet Loss für SVG-Feature-Generierung
    """
    svg_features = model_outputs['svg_features']
    target_features = targets
    
    # MSE Loss zwischen generierten und Ziel-Features
    mse_loss = F.mse_loss(svg_features, target_features)
    
    # Diversity Loss - verhindert identische Outputs
    batch_size = svg_features.size(0)
    if batch_size > 1:
        # Berechne Paarweise Distanzen zwischen Samples
        distances = torch.cdist(svg_features, svg_features, p=2)
        # Entferne Diagonale (Distanz zu sich selbst)
        mask = ~torch.eye(batch_size, dtype=bool, device=svg_features.device)
        avg_distance = distances[mask].mean()
        # Diversity Loss - belohne größere Distanzen
        diversity_loss = torch.exp(-avg_distance)
    else:
        diversity_loss = torch.tensor(0.0, device=svg_features.device)
    
    total_loss = mse_loss + alpha * diversity_loss
    
    return {
        'total_loss': total_loss,
        'mse_loss': mse_loss,
        'diversity_loss': diversity_loss
    }


class EarlyStopping:
    """Early Stopping Implementierung"""
    
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        """
        Prüft ob Early Stopping ausgelöst werden soll
        
        Returns:
            True wenn Training gestoppt werden soll, False sonst
        """
        if val_loss < self.best_loss - self.min_delta:
            # Verbesserung gefunden
            self.best_loss = val_loss
            self.counter = 0
            # Speichere bestes Modell
            self.best_model_state = model.state_dict().copy()
            return False
        else:
            # Keine Verbesserung
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early Stopping: Keine Verbesserung seit {self.patience} Epochen")
                return True
            return False
    
    def get_best_model_state(self):
        """Gibt die besten Model-Weights zurück"""
        return self.best_model_state


def create_train_val_split(dataset, val_ratio=0.1, seed=42):
    """
    Teilt das Dataset in Training und Validation auf
    
    Args:
        dataset: Das vollständige Dataset
        val_ratio: Anteil für Validation (Standard: 0.1 = 10%)
        seed: Random seed für reproduzierbare Splits
    
    Returns:
        train_dataset, val_dataset
    """
    # Seed setzen für reproduzierbare Ergebnisse
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Berechne Split-Größen
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    # Zufälliger Split
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    logger.info(f"Dataset aufgeteilt: {train_size} Training, {val_size} Validation samples")
    
    return train_dataset, val_dataset


def train_model(config):
    """Haupttrainings-Loop mit Early Stopping"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training auf: {device}")
    
    # Early Stopping initialisieren
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
    # Tokenizer initialisieren  
    try:
        text_tokenizer = AutoTokenizer.from_pretrained('gpt2')
        text_tokenizer.pad_token = text_tokenizer.eos_token
    except ImportError:
        logger.error("Transformers nicht installiert. Führe aus: pip install transformers")
        return
    
    # Dataset und DataLoader
    full_dataset = OpenMojiDataset(
        data_dir=config.data_dir,
        text_tokenizer=text_tokenizer,
        max_text_length=config.max_text_length
    )
    
    # Dataset aufteilen (konfigurierbar, Standard: 90% Training, 10% Validation)
    train_dataset, val_dataset = create_train_val_split(full_dataset, val_ratio=config.val_split)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Modell initialisieren
    model = SVGParameterModel(
        text_vocab_size=text_tokenizer.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_text_length=config.max_text_length,
        dropout=config.dropout,
        max_shapes=config.max_shapes
    ).to(device)
    
    # Optimizer und Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler - verwende ReduceLROnPlateau da wir keine feste Epochenzahl haben
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True,
        min_lr=config.learning_rate * 0.01
    )
    
    # Loss Funktionen für verschiedene Parameter
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    # Training Loop
    model.train()
    global_step = 0
    epoch = 0
    
    while True:  # Endlos-Schleife bis Early Stopping
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            text_tokens = batch['text_tokens'].to(device)
            svg_params = batch['svg_params'].to(device)  # Ground truth parameter [batch_size, 13, max_shapes]
            
            # Text-Mask erstellen (nur Padding)
            text_mask = (text_tokens == text_tokenizer.pad_token_id)
            
            # Forward pass
            optimizer.zero_grad()
            
            outputs = model(text_tokens, text_mask)
            
            # Loss berechnen - einfacher MSE für Feature-Matching
            loss_result = calculate_loss(outputs, svg_params)
            total_loss = loss_result['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            optimizer.step()
            
            # Logging
            epoch_loss += total_loss.item()
            global_step += 1
            
            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'MSE': f"{loss_result['mse_loss'].item():.4f}",
                'Div': f"{loss_result['diversity_loss'].item():.4f}",
                'Avg Loss': f"{epoch_loss/(batch_idx+1):.4f}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })
            
            # Validation und Sampling
            if global_step % config.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    # Generiere Sample
                    sample_text = text_tokens[0:1]  # Erster Text im Batch
                    sample_mask = text_mask[0:1] if text_mask is not None else None
                    generated_svgs = model.generate_svg(sample_text, sample_mask)
                    
                    original_text = batch['text'][0]
                    
                    logger.info(f"\nText: {original_text}")
                    logger.info(f"Generated SVG: {generated_svgs[0]}")
                
                model.train()
        
        # Validation nach jeder Epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_progress = tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}")
            for val_batch in val_progress:
                val_text_tokens = val_batch['text_tokens'].to(device)
                val_svg_params = val_batch['svg_params'].to(device)
                
                # Text-Mask erstellen
                val_text_mask = (val_text_tokens == text_tokenizer.pad_token_id)
                
                # Forward pass
                val_outputs = model(val_text_tokens, val_text_mask)
                
                # Loss berechnen
                val_loss_result = calculate_loss(val_outputs, val_svg_params)
                val_loss += val_loss_result['total_loss'].item()
                
                val_progress.set_postfix({'Val Loss': f"{val_loss_result['total_loss'].item():.4f}"})
        
        avg_val_loss = val_loss / len(val_dataloader)
        avg_train_loss = epoch_loss / len(train_dataloader)
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early Stopping Check
        if early_stopping(avg_val_loss, model):
            logger.info("Training beendet durch Early Stopping")
            break
        
        model.train()
        
        # Scheduler step - gib validation loss weiter
        scheduler.step(avg_val_loss)
        
        # Aktuelle Epoch erhöhen
        epoch += 1
    
    # Bestes Modell speichern
    if early_stopping.get_best_model_state() is not None:
        # Lade die besten Weights
        model.load_state_dict(early_stopping.get_best_model_state())
        
        # Speichere bestes Modell
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        best_model_path = Path(config.output_dir) / "best_model.pt"
        
        torch.save({
            'model_state_dict': early_stopping.get_best_model_state(),
            'text_tokenizer': text_tokenizer,
            'config': config,
            'best_val_loss': early_stopping.best_loss
        }, best_model_path)
        
        logger.info(f"Bestes Modell gespeichert: {best_model_path}")
        logger.info(f"Beste Validation Loss: {early_stopping.best_loss:.4f}")
    else:
        logger.warning("Kein bestes Modell gefunden!")


class TrainingConfig:
    """Konfiguration für das Training"""
    
    def __init__(self):
        # Daten
        self.data_dir = "./dataset"  # Angepasst für lokales Dataset
        self.output_dir = "./outputs"
        self.val_split = 0.1       # 10% für Validation
        
        # Modell (ressourcenschonend)
        self.d_model = 256         # Kleiner
        self.num_heads = 4         # Weniger Heads
        self.num_layers = 3        # Weniger Layer
        self.dropout = 0.1
        self.max_shapes = 8        # Max Shapes pro SVG
        
        # Text
        self.max_text_length = 64  # Kürzere Sequenzen
        
        # Training
        self.batch_size = 16       # Kann größer sein da Modell kleiner
        self.learning_rate = 2e-4  # Etwas höher
        self.weight_decay = 0.01
        self.grad_clip = 1.0
        
        # Logging und Evaluation
        self.log_interval = 50
        self.eval_interval = 200
        
        # System
        self.num_workers = 2       # Weniger Workers


def main():
    """Hauptfunktion - Einfach starten mit python train.py"""
    
    # Konfiguration erstellen
    config = TrainingConfig()
    
    # Training starten
    logger.info("Starte Text-to-SVG Training mit Early Stopping...")
    logger.info(f"Dataset: {config.data_dir}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Validation Split: {config.val_split*100}%")
    logger.info("Training läuft bis Early Stopping (5 Epochen ohne Verbesserung)")
    
    train_model(config)


if __name__ == "__main__":
    main()
