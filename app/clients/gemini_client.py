from google import genai
from google.genai import types

from app import config


def generate(prompt, system=None, model=config.DEFAULT_GEMINI_MODEL):
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=65536,
            temperature=1.0,
        ),
    )
    return response.text
