from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types
import os
import io
from datetime import datetime

load_dotenv()

app = Flask(__name__)

model = "gemini-2.5-flash-lite"
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class Output_structure(BaseModel):
    svg_code: str

system_prompt = """You are an expert SVG artist specializing in creating detailed, colorful illustrations. Your task is to generate clean, professional SVG graphics based on user descriptions.

CORE STYLE REQUIREMENTS:
- Create colorful, filled SVG graphics with appropriate colors
- Use both filled shapes and strokes as needed for realistic appearance
- Apply natural, realistic colors that match the described objects
- Use stroke-width="1-3" for outlines when needed
- All drawings must be recognizable and proportionally accurate
- ViewBox must be "0 0 64 64" - utilize the full canvas effectively

COLOR AND FILL GUIDELINES:
- Use realistic colors: green for plants, blue for water, brown for wood, etc.
- Apply gradients and multiple colors when appropriate
- Fill shapes with solid colors, gradients, or patterns as suitable
- Add outlines only when they enhance the visual clarity
- Consider shadows and highlights for depth when relevant

QUALITY STANDARDS:
- Draw exactly what the user describes - be literal and accurate
- Create realistic proportions that match real-world objects
- Include characteristic details that make objects immediately identifiable
- Use proper perspective and spatial relationships
- Ensure clean, smooth shapes with appropriate curves
- Layer elements logically from background to foreground

TECHNICAL EXECUTION:
- Start with <?xml version="1.0" encoding="UTF-8"?>
- Use proper SVG namespace: xmlns="http://www.w3.org/2000/svg"
- Employ appropriate SVG elements: <rect>, <circle>, <ellipse>, <path>, <line>, <polyline>
- Create complex shapes using <path> with proper curve commands
- Use gradients with <defs> and <linearGradient> or <radialGradient> when appropriate
- Maintain consistent coordinate system within 0-64 range
- Use meaningful comments to organize your code

STRUCTURAL APPROACH:
1. Begin with the main recognizable silhouette with base colors
2. Add primary structural components and major features with appropriate fills
3. Include secondary details that enhance recognition
4. Add fine details, textures, and color variations for realism
5. Ensure the final result is immediately identifiable as the requested object

ACCURACY IMPERATIVE:
The user's description is your blueprint. Create exactly what they describe with all characteristic features, proper colors, and realistic appearance. Always prioritize accuracy and recognition over artistic interpretation."""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_svg():
    try:
        user_input = request.json.get('prompt', '').strip()
        
        # Limit prompt to 100 characters
        if len(user_input) > 100:
            user_input = user_input[:100]
        
        if not user_input:
            return jsonify({'error': 'Please enter a description'}), 400
        
        response = client.models.generate_content(
            model=model,
            contents=user_input,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=Output_structure.model_json_schema()
            )
        )
        
        svg_code = Output_structure.model_validate(response.parsed).svg_code
        
        return jsonify({
            'svg': svg_code,
            'success': True
        })
        
    except Exception as e:
        print(f"Error generating SVG: {e}")
        return jsonify({'error': 'Failed to generate SVG. Please try again.'}), 500

@app.route('/download', methods=['POST'])
def download_svg():
    try:
        svg_content = request.json.get('svg', '')
        filename = request.json.get('filename', 'generated')
        
        if not svg_content:
            return jsonify({'error': 'No SVG content provided'}), 400
        
        # Create a safe filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_filename = safe_filename.replace(' ', '_')[:50]
        if not safe_filename:
            safe_filename = f"svg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Write SVG to BytesIO stream
        svg_io = io.BytesIO()
        svg_io.write(svg_content.encode('utf-8'))
        svg_io.seek(0)
        
        return send_file(
            svg_io,
            as_attachment=True,
            download_name=f"{safe_filename}.svg",
            mimetype='image/svg+xml'
        )
        
    except Exception as e:
        print(f"Error downloading SVG: {e}")
        return jsonify({'error': 'Failed to download SVG'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)