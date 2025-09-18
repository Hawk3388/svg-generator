"""
Sequence-to-Sequence Transformer Model für SVG-Generierung
Funktioniert wie GPT, aber für SVG-Token statt Text
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List
from svg_tokenizer import SVGTokenizer


class PositionalEncoding(nn.Module):
    """Standard Positional Encoding für Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Expand mask to match attention heads if needed
            if mask.dim() == 2:  # [batch_size, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            elif mask.dim() == 3:  # [batch_size, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
            
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations and reshape
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and put through final linear layer
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        return self.w_o(attention)


class CrossAttention(nn.Module):
    """Cross-Attention for decoder attending to encoder"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # For cross-attention: mask should be [batch_size, 1, seq_len_q, seq_len_k]
            # where seq_len_k is the encoder sequence length
            if mask.dim() == 2:  # [batch_size, seq_len_k] - encoder padding mask
                # Expand to [batch_size, 1, 1, seq_len_k] and then broadcast
                mask = mask.unsqueeze(1).unsqueeze(1)
                # Expand to match Q sequence length: [batch_size, 1, seq_len_q, seq_len_k]
                mask = mask.expand(-1, -1, Q.size(2), -1)
            
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V)
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        query: decoder hidden states [batch_size, seq_len_q, d_model]
        key_value: encoder hidden states [batch_size, seq_len_k, d_model]
        mask: encoder padding mask [batch_size, seq_len_k]
        """
        batch_size, seq_len_q, d_model = query.size()
        seq_len_k = key_value.size(1)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key_value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(key_value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply cross-attention
        attention = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and put through final linear layer
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, d_model)
        
        return self.w_o(attention)


class FeedForward(nn.Module):
    """Feed Forward Network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single Transformer Block"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x


class SVGTransformer(nn.Module):
    """
    Transformer Model für SVG-Generierung.
    Encoder: Text-Prompt -> Verstehen was generiert werden soll
    Decoder: SVG-Token generieren basierend auf Text-Eingabe
    """
    
    def __init__(self, 
                 text_vocab_size: int,
                 svg_vocab_size: int,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 d_ff: int = 2048,
                 max_text_len: int = 128,
                 max_svg_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_text_len = max_text_len
        self.max_svg_len = max_svg_len
        
        # Text Encoder
        self.text_embedding = nn.Embedding(text_vocab_size, d_model)
        self.text_pos_encoding = PositionalEncoding(d_model, max_text_len)
        
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # SVG Decoder
        self.svg_embedding = nn.Embedding(svg_vocab_size, d_model)
        self.svg_pos_encoding = PositionalEncoding(d_model, max_svg_len)
        
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Cross-Attention Layers (Decoder attending to Encoder)
        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(d_model, num_heads, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        self.cross_attention_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_decoder_layers)
        ])
        
        # Output Layer
        self.output_projection = nn.Linear(d_model, svg_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_padding_mask(self, tokens: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        """Create padding mask"""
        return (tokens != pad_token_id).unsqueeze(1).unsqueeze(2)
    
    def create_causal_mask(self, size: int) -> torch.Tensor:
        """Create causal (triangular) mask for decoder"""
        mask = torch.tril(torch.ones(size, size))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def encode_text(self, text_tokens: torch.Tensor, text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text input"""
        # Embedding + Positional Encoding
        x = self.text_embedding(text_tokens) * math.sqrt(self.d_model)
        x = self.text_pos_encoding(x)
        x = self.dropout(x)
        
        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, text_mask)
        
        return x
    
    def decode_svg(self, svg_tokens: torch.Tensor, encoded_text: torch.Tensor,
                   svg_mask: Optional[torch.Tensor] = None, 
                   text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode SVG tokens"""
        # Embedding + Positional Encoding
        x = self.svg_embedding(svg_tokens) * math.sqrt(self.d_model)
        x = self.svg_pos_encoding(x)
        x = self.dropout(x)
        
        # Decoder layers with cross-attention
        for i, (decoder_layer, cross_attn, cross_norm) in enumerate(
            zip(self.decoder_layers, self.cross_attention_layers, self.cross_attention_norms)
        ):
            # Self-attention (causal)
            x = decoder_layer(x, svg_mask)
            
            # Cross-attention to encoded text
            cross_attn_output = cross_attn(x, encoded_text, text_mask)  # Q from decoder, K,V from encoder
            x = cross_norm(x + self.dropout(cross_attn_output))
        
        return x
    
    def forward(self, text_tokens: torch.Tensor, svg_tokens: torch.Tensor,
                text_pad_token_id: int = 0, svg_pad_token_id: int = 0) -> torch.Tensor:
        """
        Forward pass
        text_tokens: [batch_size, text_seq_len]
        svg_tokens: [batch_size, svg_seq_len]
        """
        batch_size, text_len = text_tokens.size()
        svg_len = svg_tokens.size(1)
        
        # Create masks
        text_mask = self.create_padding_mask(text_tokens, text_pad_token_id)
        svg_padding_mask = self.create_padding_mask(svg_tokens, svg_pad_token_id)
        svg_causal_mask = self.create_causal_mask(svg_len).to(svg_tokens.device)
        
        # Combine masks properly (both should be boolean)
        svg_mask = svg_padding_mask.bool() & svg_causal_mask.bool()
        
        # Encode text
        encoded_text = self.encode_text(text_tokens, text_mask)
        
        # Decode SVG
        decoded_svg = self.decode_svg(svg_tokens, encoded_text, svg_mask, text_mask)
        
        # Project to vocabulary
        logits = self.output_projection(decoded_svg)
        
        return logits
    
    def generate_svg(self, text_tokens: torch.Tensor, svg_tokenizer: SVGTokenizer,
                     max_length: int = 512, temperature: float = 1.0,
                     top_k: int = 50, top_p: float = 0.9) -> List[int]:
        """
        Generate SVG tokens autoregressively
        """
        self.eval()
        device = text_tokens.device
        batch_size = text_tokens.size(0)
        
        # Encode text once
        with torch.no_grad():
            text_mask = self.create_padding_mask(text_tokens)
            encoded_text = self.encode_text(text_tokens, text_mask)
        
        # Start with SOS token
        sos_token_id = svg_tokenizer.token_to_id['<SOS>']
        eos_token_id = svg_tokenizer.token_to_id['<EOS>']
        
        generated_tokens = [sos_token_id]
        
        for _ in range(max_length):
            # Prepare input
            svg_input = torch.tensor([generated_tokens], device=device)
            
            with torch.no_grad():
                # Create causal mask
                svg_len = svg_input.size(1)
                svg_mask = self.create_causal_mask(svg_len).to(device)
                
                # Forward pass
                decoded = self.decode_svg(svg_input, encoded_text, svg_mask, text_mask)
                logits = self.output_projection(decoded)
                
                # Get logits for last position
                last_logits = logits[0, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                next_token_id = self._sample_token(last_logits, top_k, top_p)
                
                generated_tokens.append(next_token_id)
                
                # Check for EOS
                if next_token_id == eos_token_id:
                    break
        
        return generated_tokens
    
    def _sample_token(self, logits: torch.Tensor, top_k: int, top_p: float) -> int:
        """Sample next token with top-k and top-p filtering"""
        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(0, top_k_indices, top_k_logits)
        
        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        
        return next_token_id


class SVGGenerationModel(nn.Module):
    """
    Wrapper Model für Training und Inference
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize tokenizers (will be set during training)
        self.text_tokenizer = None
        self.svg_tokenizer = None
        
        # Model will be initialized after tokenizers are ready
        self.transformer = None
    
    def initialize_model(self, text_vocab_size: int, svg_vocab_size: int):
        """Initialize transformer after vocab sizes are known"""
        self.transformer = SVGTransformer(
            text_vocab_size=text_vocab_size,
            svg_vocab_size=svg_vocab_size,
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            num_encoder_layers=self.config.num_encoder_layers,
            num_decoder_layers=self.config.num_decoder_layers,
            d_ff=self.config.d_ff,
            max_text_len=self.config.max_text_len,
            max_svg_len=self.config.max_svg_len,
            dropout=self.config.dropout
        )
    
    def forward(self, text_tokens: torch.Tensor, svg_tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass for training"""
        if self.transformer is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        return self.transformer(text_tokens, svg_tokens)
    
    def generate(self, text_prompt: str, max_length: int = 512) -> str:
        """Generate SVG from text prompt"""
        if self.transformer is None or self.text_tokenizer is None or self.svg_tokenizer is None:
            raise RuntimeError("Model or tokenizers not initialized.")
        
        # Tokenize text
        text_tokens = self.text_tokenizer.encode(
            text_prompt, 
            max_length=self.config.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Generate SVG tokens
        svg_token_ids = self.transformer.generate_svg(
            text_tokens, 
            self.svg_tokenizer, 
            max_length=max_length
        )
        
        # Convert back to SVG string
        svg_string = self.svg_tokenizer.detokenize(svg_token_ids)
        
        return svg_string