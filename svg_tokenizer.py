"""
Robuster SVG-Tokenizer für Sequence-to-Sequence SVG-Generierung
Behandelt SVG-Elemente als strukturierte Token-Sequenzen
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Union
import xml.etree.ElementTree as ET
from pathlib import Path


class SVGTokenizer:
    """
    Tokenizer der SVG-Inhalte in eine Token-Sequenz umwandelt.
    Funktioniert ähnlich wie Textgenerierung, aber für SVG-Strukturen.
    """
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.is_frozen = False  # Add freeze mode for training
        
        # Spezielle Tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<SOS>': 1,  # Start of SVG
            '<EOS>': 2,  # End of SVG  
            '<UNK>': 3,  # Unknown
            '<SEP>': 4,  # Separator zwischen Attributen
        }
        
        # SVG-Element Tokens
        self.svg_elements = [
            'svg', 'g', 'path', 'circle', 'rect', 'ellipse', 'line', 
            'polygon', 'polyline', 'text', 'tspan', 'defs', 'use',
            'clipPath', 'mask', 'image'
        ]
        
        # SVG-Attribut Tokens
        self.svg_attributes = [
            'id', 'class', 'viewBox', 'xmlns', 'width', 'height',
            'd', 'cx', 'cy', 'r', 'rx', 'ry', 'x', 'y', 'x1', 'y1', 'x2', 'y2',
            'fill', 'stroke', 'stroke-width', 'stroke-linecap', 'stroke-linejoin',
            'transform', 'opacity', 'style'
        ]
        
        # Path-Kommando Tokens
        self.path_commands = [
            'M', 'L', 'H', 'V', 'C', 'S', 'Q', 'T', 'A', 'Z',
            'm', 'l', 'h', 'v', 'c', 's', 'q', 't', 'a', 'z'
        ]
        
        # Numerische Precision für Koordinaten
        self.coord_precision = 2
        
        # Build initial vocabulary
        self._build_base_vocabulary()
    
    def _build_base_vocabulary(self):
        """Erstelle Basis-Vokabular mit festen SVG-Tokens"""
        vocab = []
        
        # Spezielle Tokens
        vocab.extend(self.special_tokens.keys())
        
        # Element Start/End Tags
        for elem in self.svg_elements:
            vocab.extend([f'<{elem}>', f'</{elem}>'])
        
        # Attribute Namen
        for attr in self.svg_attributes:
            vocab.append(f'{attr}=')
        
        # Path Kommandos
        vocab.extend(self.path_commands)
        
        # Häufige Werte
        vocab.extend(['none', 'round', '#000000', '#ffffff', 'black', 'white'])
        
        # Erstelle Mapping
        for i, token in enumerate(vocab):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        self.next_id = len(vocab)
    
    def add_token(self, token: str) -> int:
        """Füge neuen Token hinzu"""
        if token not in self.token_to_id:
            # If frozen (during training), return UNK for unknown tokens
            if self.is_frozen:
                return self.token_to_id['<UNK>']
                
            if self.next_id >= self.vocab_size:
                return self.token_to_id['<UNK>']
            
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        
        return self.token_to_id[token]
    
    def freeze_vocab(self):
        """Freeze vocabulary - no new tokens will be added"""
        self.is_frozen = True
        
    def unfreeze_vocab(self):
        """Unfreeze vocabulary - allow adding new tokens"""
        self.is_frozen = False
    
    def parse_svg_to_tokens(self, svg_content: str) -> List[str]:
        """
        Konvertiert SVG-Inhalt zu Token-Sequenz.
        Jedes Element wird zu einer Sequenz von Tokens.
        """
        tokens = ['<SOS>']
        
        try:
            # Parse SVG
            root = ET.fromstring(svg_content)
            tokens.extend(self._element_to_tokens(root))
            
        except ET.ParseError as e:
            print(f"SVG Parse Error: {e}")
            return ['<SOS>', '<UNK>', '<EOS>']
        
        tokens.append('<EOS>')
        return tokens
    
    def _element_to_tokens(self, element: ET.Element) -> List[str]:
        """Konvertiert XML-Element zu Token-Sequenz"""
        tokens = []
        
        # Element Start Tag
        tag_name = element.tag.split('}')[-1]  # Remove namespace
        tokens.append(f'<{tag_name}>')
        
        # Attribute
        for attr_name, attr_value in element.attrib.items():
            attr_name = attr_name.split('}')[-1]  # Remove namespace
            tokens.append(f'{attr_name}=')
            
            # Tokenisiere Attributwert
            if attr_name == 'd':  # Path data
                tokens.extend(self._tokenize_path_data(attr_value))
            elif attr_name in ['cx', 'cy', 'r', 'rx', 'ry', 'x', 'y', 'width', 'height']:
                tokens.append(self._normalize_number(attr_value))
            else:
                tokens.append(attr_value)
            
            tokens.append('<SEP>')
        
        # Element Inhalt (Text)
        if element.text and element.text.strip():
            tokens.append(element.text.strip())
        
        # Kinder-Elemente
        for child in element:
            tokens.extend(self._element_to_tokens(child))
        
        # Element End Tag
        tokens.append(f'</{tag_name}>')
        
        return tokens
    
    def _tokenize_path_data(self, path_data: str) -> List[str]:
        """Tokenisiert SVG-Path-Daten"""
        tokens = []
        
        # Split path in Kommandos und Koordinaten
        # Regex pattern for path commands and coordinates
        pattern = r'([MmLlHhVvCcSsQqTtAaZz])|(-?\d*\.?\d+)'
        
        matches = re.findall(pattern, path_data)
        
        for match in matches:
            command, number = match
            if command:
                tokens.append(command)
            elif number:
                tokens.append(self._normalize_number(number))
        
        return tokens
    
    def _normalize_number(self, num_str: str) -> str:
        """Normalisiert numerische Werte"""
        try:
            num = float(num_str)
            return str(round(num, self.coord_precision))
        except ValueError:
            return num_str
    
    def tokenize(self, svg_content: str) -> List[int]:
        """
        Hauptfunktion: SVG zu Token-IDs
        """
        tokens = self.parse_svg_to_tokens(svg_content)
        
        # Konvertiere zu IDs
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # Neuen Token hinzufügen
                token_id = self.add_token(token)
                token_ids.append(token_id)
        
        return token_ids
    
    def detokenize(self, token_ids: List[int]) -> str:
        """
        Token-IDs zurück zu SVG-String
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append('<UNK>')
        
        return self._tokens_to_svg(tokens)
    
    def _tokens_to_svg(self, tokens: List[str]) -> str:
        """Rekonstruiert SVG aus Token-Sequenz"""
        svg_parts = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            if token == '<SOS>':
                i += 1
                continue
            elif token == '<EOS>':
                break
            elif token.startswith('<') and token.endswith('>') and not token.startswith('</'):
                # Start Tag
                tag_name = token[1:-1]
                element_str, i = self._build_element_string(tokens, i, tag_name)
                svg_parts.append(element_str)
            else:
                i += 1
        
        return '\\n'.join(svg_parts)
    
    def _build_element_string(self, tokens: List[str], start_idx: int, tag_name: str) -> Tuple[str, int]:
        """Baut Element-String aus Tokens"""
        i = start_idx + 1  # Skip start tag
        attributes = []
        content = []
        
        # Parse Attribute
        while i < len(tokens):
            token = tokens[i]
            
            if token.endswith('='):
                attr_name = token[:-1]
                i += 1
                if i < len(tokens):
                    attr_value = tokens[i]
                    if attr_name == 'd':
                        # Baue Path-Daten
                        path_data = []
                        i += 1
                        while i < len(tokens) and tokens[i] != '<SEP>' and not tokens[i].startswith('<'):
                            path_data.append(tokens[i])
                            i += 1
                        attr_value = ' '.join(path_data)
                        i -= 1  # Zurück für nächste Iteration
                    
                    attributes.append(f'{attr_name}="{attr_value}"')
                i += 1
            elif token == '<SEP>':
                i += 1
                continue
            elif token.startswith('<') and not token.startswith('</'):
                # Kinder-Element
                child_tag = token[1:-1]
                child_str, i = self._build_element_string(tokens, i, child_tag)
                content.append(child_str)
            elif token == f'</{tag_name}>':
                break
            else:
                # Text Content
                content.append(token)
                i += 1
        
        # Baue finalen Element-String
        attr_str = ' '.join(attributes)
        if attr_str:
            attr_str = ' ' + attr_str
        
        if content:
            content_str = '\\n'.join(content)
            element = f'<{tag_name}{attr_str}>\\n{content_str}\\n</{tag_name}>'
        else:
            element = f'<{tag_name}{attr_str}/>'
        
        return element, i + 1
    
    def get_vocab_size(self) -> int:
        """Gibt aktuelle Vokabular-Größe zurück"""
        return self.next_id
    
    def save_vocab(self, filepath: str):
        """Speichere Vokabular"""
        # Erstelle Ordner falls nicht vorhanden
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        vocab_data = {
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'next_id': self.next_id,
            'vocab_size': self.vocab_size
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    
    def load_vocab(self, filepath: str):
        """Lade Vokabular"""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data['token_to_id']
        # Convert string keys back to int for id_to_token
        self.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
        self.next_id = vocab_data['next_id']
        self.vocab_size = vocab_data['vocab_size']


def test_svg_tokenizer():
    """Test-Funktion für den SVG-Tokenizer"""
    
    # Test SVG
    test_svg = '''<svg viewBox="0 0 72 72" xmlns="http://www.w3.org/2000/svg">
  <circle cx="36" cy="36" r="23" fill="none" stroke="#000000"/>
  <path d="M30,31c0,1.66-1.34,3-3,3s-3-1.34-3-3c0-1.66,1.34-3,3-3S30,29.34,30,31"/>
</svg>'''
    
    tokenizer = SVGTokenizer()
    
    print("Original SVG:")
    print(test_svg)
    print("\\n" + "="*50 + "\\n")
    
    # Tokenisierung
    tokens = tokenizer.parse_svg_to_tokens(test_svg)
    print("Tokens:")
    print(tokens)
    print("\\n" + "="*50 + "\\n")
    
    # Token IDs
    token_ids = tokenizer.tokenize(test_svg)
    print("Token IDs:")
    print(token_ids)
    print("\\n" + "="*50 + "\\n")
    
    # Detokenisierung
    reconstructed = tokenizer.detokenize(token_ids)
    print("Reconstructed SVG:")
    print(reconstructed)
    
    print(f"\\nVocabulary size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    test_svg_tokenizer()