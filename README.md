# Text-to-SVG Generator mit PyTorch

Dieses Projekt implementiert ein Transformer-basiertes Modell zur Generierung von SVG-Dateien aus Text-Prompts unter Verwendung des OpenMoji-Datasets.

## Features

- **Transformer-Architektur**: Encoder-Decoder-Modell mit Multi-Head Attention
- **Spezieller SVG-Tokenizer**: Optimiert für SVG-Syntax und -Struktur
- **OpenMoji-Dataset**: Automatischer Download und Verarbeitung von Emoji-SVGs
- **Flexibles Training**: Konfigurierbare Hyperparameter und Logging
- **Wandb-Integration**: Optionales Experiment-Tracking

## Installation

1. Klonen Sie das Repository oder laden Sie die Dateien herunter
2. Installieren Sie die Abhängigkeiten:

```bash
pip install -r requirements.txt
```

### Systemanforderungen

- Python 3.8+
- CUDA-fähige GPU (empfohlen, aber nicht erforderlich)
- Mindestens 8GB RAM
- 2GB freier Festplattenspeicher für Dataset

## Schnellstart

### Training starten

```bash
# Einfaches Training mit Standardparametern
python train.py

# Mit spezifischen Parametern
python train.py --batch_size 16 --num_epochs 100 --learning_rate 2e-4

# Mit Wandb-Logging
python train.py --use_wandb --batch_size 8 --num_epochs 50
```

### Parameter

| Parameter | Beschreibung | Standard |
|-----------|--------------|----------|
| `--data_dir` | Pfad zum OpenMoji-Dataset | `./data/openmoji` |
| `--output_dir` | Ausgabeverzeichnis für Modelle | `./outputs` |
| `--batch_size` | Batch-Größe | `8` |
| `--num_epochs` | Anzahl Epochen | `50` |
| `--learning_rate` | Lernrate | `1e-4` |
| `--use_wandb` | Wandb-Logging aktivieren | `False` |

## Modell-Architektur

### Überblick

Das Modell basiert auf einer Transformer-Encoder-Decoder-Architektur:

1. **Text-Encoder**: Verarbeitet Eingabe-Text zu Kontext-Embeddings
2. **SVG-Decoder**: Generiert SVG-Tokens basierend auf Text-Kontext
3. **Cross-Attention**: Verbindet Text- und SVG-Repräsentationen

### Komponenten

- **SVGTokenizer**: Spezialisierter Tokenizer für SVG-Syntax
- **MultiHeadAttention**: Selbst- und Cross-Attention-Mechanismen
- **TransformerBlock**: Encoder/Decoder-Layer mit Normalisierung
- **TextToSVGModel**: Haupt-Modell-Klasse

### Hyperparameter

- **d_model**: 512 (Embedding-Dimension)
- **num_heads**: 8 (Attention-Köpfe)
- **num_layers**: 6 (Encoder + Decoder Layer)
- **d_ff**: 2048 (Feed-Forward-Dimension)
- **vocab_sizes**: 8000 (SVG), variable (Text)

## Dataset

Das Projekt verwendet das OpenMoji-Dataset:

- **Quelle**: [OpenMoji GitHub](https://github.com/hfg-gmuend/openmoji)
- **Format**: SVG-Dateien mit Metadaten (Beschreibungen, Tags)
- **Größe**: ~4000 Emojis (Demo-Version lädt ersten 100)
- **Download**: Automatisch beim ersten Lauf

### Dataset-Struktur

```
data/openmoji/
├── openmoji.json          # Metadaten
└── svg/                   # SVG-Dateien
    ├── 1F600.svg         # 😀
    ├── 1F601.svg         # 😁
    └── ...
```

## Training

### Trainingsprozess

1. **Daten-Loading**: OpenMoji-SVGs mit Beschreibungen
2. **Tokenization**: Text (GPT-2) + SVG (custom tokenizer)
3. **Batch-Processing**: Masking für Attention
4. **Loss-Berechnung**: Cross-Entropy mit Padding-Ignore
5. **Optimization**: AdamW mit Cosine Annealing

### Monitoring

- **Progress Bars**: Echtzeit-Loss und Metriken
- **Sampling**: Regelmäßige Generierung von Beispielen
- **Checkpoints**: Modell-Speicherung nach jeder 5. Epoche
- **Wandb**: Experiment-Tracking (optional)

### Tipps für bessere Ergebnisse

1. **GPU verwenden**: Deutlich schnelleres Training
2. **Batch-Größe anpassen**: Je nach GPU-Memory
3. **Learning Rate**: Starten Sie mit 1e-4, experimentieren Sie
4. **Längeres Training**: 100+ Epochen für bessere Qualität
5. **Dataset erweitern**: Mehr SVGs für bessere Generalisierung

## Generierung

Nach dem Training können Sie SVGs generieren:

```python
import torch
from train import TextToSVGModel, SVGTokenizer

# Modell laden
checkpoint = torch.load('outputs/final_model.pt')
model = TextToSVGModel(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
svg_tokenizer = checkpoint['svg_tokenizer']

# Text tokenisieren (vereinfacht)
text = "smiling face emoji"
# ... tokenization logic ...

# SVG generieren
model.eval()
with torch.no_grad():
    generated_svg = model.generate(text_tokens, max_length=512)
    svg_string = svg_tokenizer.decode(generated_svg[0].tolist())

print(svg_string)
```

## Troubleshooting

### Häufige Probleme

**Memory-Fehler**: Reduzieren Sie `batch_size` oder `max_svg_length`

```bash
python train.py --batch_size 4 --max_svg_length 256
```

**Slow Training**: GPU-Support prüfen

```python
import torch
print(torch.cuda.is_available())  # Sollte True sein
```

**Dataset-Download-Fehler**: Manuelle OpenMoji-Installation

```bash
mkdir -p data/openmoji
wget https://github.com/hfg-gmuend/openmoji/archive/master.zip
```

**Schlechte Qualität**: Längeres Training oder größeres Modell

```bash
python train.py --num_epochs 200 --d_model 768 --num_heads 12
```

## Erweiterte Nutzung

### Custom Dataset

Erstellen Sie Ihre eigene `Dataset`-Klasse:

```python
class CustomSVGDataset(Dataset):
    def __init__(self, svg_files, descriptions):
        # Ihre Implementierung
        pass
```

### Modell-Anpassungen

Ändern Sie die Architektur in `TextToSVGModel`:

```python
# Mehr Layer
num_encoder_layers=12
num_decoder_layers=12

# Größere Embeddings
d_model=1024
d_ff=4096
```

### Transfer Learning

Nutzen Sie vortrainierte Text-Encoder:

```python
from transformers import AutoModel
text_encoder = AutoModel.from_pretrained('bert-base-uncased')
```

## Evaluation

### Metriken

- **Perplexity**: Modell-Unsicherheit
- **BLEU Score**: Sequenz-Ähnlichkeit (experimentell für SVG)
- **Manual Inspection**: Visuelle Qualität der generierten SVGs

### Validation

Implementieren Sie Validierungs-Loop:

```python
def validate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            # ... validation logic ...
    return total_loss / len(dataloader)
```

## Contributing

1. Fork das Repository
2. Erstellen Sie einen Feature-Branch
3. Committen Sie Ihre Änderungen
4. Erstellen Sie einen Pull Request

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Das OpenMoji-Dataset hat seine eigene [CC BY-SA 4.0 Lizenz](https://github.com/hfg-gmuend/openmoji/blob/master/LICENSE.txt).

## Referenzen

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer Paper
- [OpenMoji](https://openmoji.org/) - Open Source Emoji Dataset
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

## Support

Bei Fragen oder Problemen erstellen Sie ein Issue oder kontaktieren Sie den Entwickler.