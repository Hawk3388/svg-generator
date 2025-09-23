from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types
import os

load_dotenv()

model = "gemini-2.5-flash"
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class Output_structure(BaseModel):
    svg_code: str

def main():
    system_prompt = """You are an expert SVG line artist specializing in creating precise, detailed line drawings. Your task is to generate clean, professional SVG illustrations based on user descriptions.

CORE STYLE REQUIREMENTS:
- Create ONLY black outline drawings using stroke="black" and fill="none"
- Never use filled shapes - pure line art only
- Use stroke-width="2" for main structural outlines
- Use stroke-width="1.5" for fine details and internal elements
- All drawings must be recognizable and proportionally accurate
- ViewBox must be "0 0 64 64" - utilize the full canvas effectively

QUALITY STANDARDS:
- Draw exactly what the user describes - be literal and accurate
- Create realistic proportions that match real-world objects
- Include characteristic details that make objects immediately identifiable
- Use proper perspective and spatial relationships
- Ensure clean, smooth lines with appropriate curves
- Layer elements logically from background to foreground

TECHNICAL EXECUTION:
- Start with <?xml version="1.0" encoding="UTF-8"?>
- Use proper SVG namespace: xmlns="http://www.w3.org/2000/svg"
- Employ appropriate SVG elements: <rect>, <circle>, <ellipse>, <path>, <line>, <polyline>
- Create complex shapes using <path> with proper curve commands
- Maintain consistent coordinate system within 0-64 range
- Use meaningful comments to organize your code

STRUCTURAL APPROACH:
1. Begin with the main recognizable silhouette or outline
2. Add primary structural components and major features  
3. Include secondary details that enhance recognition
4. Add fine details and textures through additional line work
5. Ensure the final result is immediately identifiable as the requested object

ACCURACY IMPERATIVE:
The user's description is your blueprint. Create exactly what they describe with all characteristic features and proper structure. Always prioritize accuracy and recognition over artistic interpretation."""

    while True:
        user_input = input("Describe the SVG you want (or type 'quit' to quit): ")
        if user_input.lower() == 'quit':
            break

        response = client.models.generate_content(
            model=model,
            contents=user_input,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=Output_structure.model_json_schema()
            )
        )

        try:
            svg_code = Output_structure.model_validate(response.parsed).svg_code
            # Create a safe filename from user input
            safe_filename = "".join(c for c in user_input if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_filename = safe_filename.replace(' ', '_')[:50]  # Limit length and replace spaces
            if not safe_filename:
                safe_filename = "generated_svg"
            
            with open(f"./outputs/{safe_filename}.svg", "w") as f:
                f.write(svg_code)
            print(f"✅ SVG saved as: {safe_filename}.svg")
        except Exception as e:
            print("Error parsing response:", e)
            print("Raw response:", response)

if __name__ == "__main__":
    main()