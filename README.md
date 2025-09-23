# SVG Generator

A simple AI-powered SVG generator that creates SVG graphics from text descriptions.

## Features

- Generate SVG graphics from text descriptions
- Web interface with Flask
- Download function for generated SVGs
- Prompt limit of 100 characters

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

## Configuration

Create a `.env` file with your Google API Key:

```env
GOOGLE_API_KEY=your_api_key_here
```

## Deployment

The app is prepared for Render.com. Don't forget to set the `GOOGLE_API_KEY` as an environment variable.
