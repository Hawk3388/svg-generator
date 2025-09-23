# SVG Generator

A simple AI-powered SVG generator that creates SVG graphics from text descriptions.

## Features

- Generate SVG graphics from text descriptions
- Web interface with Flask
- Download function for generated SVGs
- Prompt limit of 100 characters

## Components

### AI-Powered Generation

- **`gemini_main.py`** - Command-line interface using Google Gemini AI
- **`app.py`** - Flask web application for browser-based usage

### Machine Learning Training (Experimental)

- **`train.py`** - Training script for sequence-to-sequence SVG generation
- **`svg_tokenizer.py`** - SVG tokenizer for converting SVG to token sequences
- **`svg_transformer.py`** - Transformer model architecture for SVG generation

### Dataset

- **`dataset/`** - OpenMoji emoji dataset with thousands of SVG files
- Contains Unicode emoji SVGs from the OpenMoji project
- Used for training custom SVG generation models

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
python gemini_main.py
```

### Web App

```bash
python app.py
```

Then open <http://localhost:5000> in your browser.

### Training (Experimental)

```bash
python train.py
```

## Configuration

Create a `.env` file with your Google API Key:

```env
GOOGLE_API_KEY=your_api_key_here
```

## Deployment

The app is prepared for Render.com. Don't forget to set the `GOOGLE_API_KEY` as an environment variable.
