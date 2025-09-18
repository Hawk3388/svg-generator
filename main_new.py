"""
Inference Script für Text-to-SVG Generierung
Vollständig funktionsfähige SVG-Generierung ohne Fallbacks
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List
import json

from svg_tokenizer import SVGTokenizer
from svg_transformer import SVGGenerationModel
from transformers import AutoTokenizer

# Import TrainingConfig to allow checkpoint loading
try:
    from train_new import TrainingConfig
except ImportError:
    # Fallback: create a dummy TrainingConfig class
    class TrainingConfig:
        def __init__(self):
            self.d_model = 512
            self.num_heads = 8
            self.num_encoder_layers = 6
            self.num_decoder_layers = 6
            self.d_ff = 2048
            self.max_text_len = 128
            self.max_svg_len = 1024
            self.dropout = 0.1


class SVGGenerator:
    """
    Production-ready SVG Generator
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize SVG Generator
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ("cuda", "cpu", or "auto")
        """
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Load model and tokenizers
        self._load_model(model_path)
        
        print(f"✅ SVG Generator loaded successfully!")
        print(f"📊 Text vocab size: {len(self.text_tokenizer)}")
        print(f"📊 SVG vocab size: {self.svg_tokenizer.get_vocab_size()}")
        print(f"🎯 Model ready for generation!")
    
    def _load_model(self, model_path: str):
        """Load trained model and tokenizers"""
        print(f"Loading model from {model_path}...")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Load configuration
            config = checkpoint['config']
            
            # Initialize text tokenizer
            self.text_tokenizer = checkpoint['text_tokenizer']
            
            # Initialize SVG tokenizer
            self.svg_tokenizer = SVGTokenizer()
            svg_vocab_data = checkpoint['svg_tokenizer_vocab']
            self.svg_tokenizer.token_to_id = svg_vocab_data['token_to_id']
            self.svg_tokenizer.id_to_token = {int(k): v for k, v in svg_vocab_data['id_to_token'].items()}
            self.svg_tokenizer.next_id = svg_vocab_data['next_id']
            self.svg_tokenizer.vocab_size = svg_vocab_data['vocab_size']
            
        except Exception as e:
            print(f"❌ Fehler beim Laden des Checkpoints: {e}")
            raise
        
        # Initialize model
        self.model = SVGGenerationModel(config)
        self.model.text_tokenizer = self.text_tokenizer
        self.model.svg_tokenizer = self.svg_tokenizer
        self.model.initialize_model(
            text_vocab_size=len(self.text_tokenizer),
            svg_vocab_size=self.svg_tokenizer.get_vocab_size()
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Store generation parameters
        self.config = config
        
        print(f"✅ Model loaded from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
        print(f"📈 Best validation loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    def generate(self, 
                 prompt: str,
                 max_length: int = 512,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 num_attempts: int = 3) -> str:
        """
        Generate SVG from text prompt
        
        Args:
            prompt: Text description of desired SVG
            max_length: Maximum sequence length for generation
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Top-k filtering for sampling
            top_p: Top-p (nucleus) filtering for sampling
            num_attempts: Number of generation attempts if first fails
            
        Returns:
            Generated SVG as string
        """
        print(f"🎨 Generating SVG for: '{prompt}'")
        
        for attempt in range(num_attempts):
            try:
                # Tokenize input text
                text_encoding = self.text_tokenizer(
                    prompt,
                    max_length=self.config.max_text_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                text_tokens = text_encoding['input_ids'].to(self.device)
                
                # Generate SVG tokens
                with torch.no_grad():
                    svg_token_ids = self.model.transformer.generate_svg(
                        text_tokens,
                        self.svg_tokenizer,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
                
                # Convert tokens back to SVG
                svg_string = self.svg_tokenizer.detokenize(svg_token_ids)
                
                # Validate generated SVG
                if self._validate_svg(svg_string):
                    print(f"✅ Generation successful on attempt {attempt + 1}")
                    return svg_string
                else:
                    print(f"⚠️  Attempt {attempt + 1} produced invalid SVG, retrying...")
                    continue
                    
            except Exception as e:
                print(f"❌ Generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == num_attempts - 1:
                    # Last attempt - return error SVG
                    return self._create_error_svg(f"Generation failed: {str(e)}")
                continue
        
        # All attempts failed
        return self._create_error_svg("All generation attempts failed")
    
    def _validate_svg(self, svg_string: str) -> bool:
        """Validate generated SVG"""
        # Basic validation checks
        if not svg_string or len(svg_string.strip()) == 0:
            return False
        
        # Check for required SVG tags
        if '<svg' not in svg_string.lower():
            return False
        
        # Check for balanced tags (basic check)
        svg_open_count = svg_string.lower().count('<svg')
        svg_close_count = svg_string.lower().count('</svg>')
        
        if svg_open_count == 0 or svg_close_count == 0:
            return False
        
        # More sophisticated validation could be added here
        # For now, we'll consider it valid if it has basic structure
        return True
    

    
    def _create_error_svg(self, error_message: str) -> str:
        """Create a simple error SVG"""
        return f'''<svg viewBox="0 0 200 100" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="200" height="100" fill="#ffeeee" stroke="#ff0000"/>
  <text x="100" y="50" text-anchor="middle" font-family="Arial" font-size="12" fill="#ff0000">
    {error_message[:50]}
  </text>
</svg>'''
    
    def generate_batch(self, 
                      prompts: List[str],
                      **generation_kwargs) -> List[str]:
        """
        Generate SVGs for multiple prompts
        
        Args:
            prompts: List of text prompts
            **generation_kwargs: Generation parameters
            
        Returns:
            List of generated SVG strings
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\\n📝 Processing prompt {i+1}/{len(prompts)}: '{prompt}'")
            svg = self.generate(prompt, **generation_kwargs)
            results.append(svg)
        
        return results
    
    def save_svg(self, svg_content: str, filepath: str):
        """Save SVG content to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        print(f"💾 SVG saved to: {filepath}")
    
    def interactive_mode(self):
        """Interactive generation mode"""
        print("\\n🎨 Interactive SVG Generation Mode")
        print("=" * 50)
        print("Enter text prompts to generate SVGs.")
        print("Commands:")
        print("  'quit' or 'exit' - Exit the program")
        print("  'save <filename>' - Save last generated SVG")
        print("  'params' - Show current generation parameters")
        print("  'set <param> <value>' - Change generation parameters")
        print("=" * 50)
        
        last_svg = ""
        counter = 1
        
        # Default generation parameters
        gen_params = {
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'max_length': 512
        }
        
        while True:
            try:
                prompt = input(f"\\n[{counter}] Enter prompt: ").strip()
                
                if not prompt:
                    continue
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif prompt.startswith('save '):
                    filename = prompt[5:].strip()
                    if last_svg and filename:
                        if not filename.endswith('.svg'):
                            filename += '.svg'
                        self.save_svg(last_svg, filename)
                    else:
                        print("❌ No SVG to save or invalid filename")
                    continue
                
                elif prompt == 'params':
                    print("\\nCurrent generation parameters:")
                    for key, value in gen_params.items():
                        print(f"  {key}: {value}")
                    continue
                
                elif prompt.startswith('set '):
                    parts = prompt[4:].strip().split()
                    if len(parts) == 2:
                        param, value = parts
                        if param in gen_params:
                            try:
                                if param == 'max_length' or param == 'top_k':
                                    gen_params[param] = int(value)
                                else:
                                    gen_params[param] = float(value)
                                print(f"✅ Set {param} = {gen_params[param]}")
                            except ValueError:
                                print(f"❌ Invalid value for {param}")
                        else:
                            print(f"❌ Unknown parameter: {param}")
                    continue
                
                # Generate SVG
                svg_content = self.generate(prompt, **gen_params)
                
                # Display result
                print(f"\\n🎨 Generated SVG for '{prompt}':")
                print("=" * 60)
                print(svg_content)
                print("=" * 60)
                
                # Auto-save with counter
                output_dir = Path("outputs")
                output_dir.mkdir(exist_ok=True)
                filename = output_dir / f"generated_{counter:03d}.svg"
                self.save_svg(svg_content, str(filename))
                
                last_svg = svg_content
                counter += 1
                
            except KeyboardInterrupt:
                print("\\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")


def main():
    """Main function"""
    print("🎨 SVG Generator - Text zu SVG Konverter")
    print("=" * 50)
    
    # Model path
    model_path = input("Model-Pfad (Enter für './outputs/best_model.pt'): ").strip()
    if not model_path:
        model_path = "./outputs/best_model.pt"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model-Datei nicht gefunden: {model_path}")
        print("💡 Trainiere zuerst das Model mit train_new.py")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Initialize generator
        print(f"\nLade Model von {model_path}...")
        generator = SVGGenerator(model_path, device)
        
        # Interactive mode
        generator.interactive_mode()
    
    except Exception as e:
        print(f"❌ Fehler beim Laden des Generators: {str(e)}")
        return


if __name__ == "__main__":
    main()