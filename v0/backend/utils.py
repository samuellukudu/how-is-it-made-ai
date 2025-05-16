from google import genai
from google.genai import types
from typing import Union, List, Generator, Dict
from PIL import Image
from io import BytesIO
import base64
import requests
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(
    api_key=os.getenv("API_KEY")
)

def bytes_to_base64(data: bytes, with_prefix: bool = True) -> str:
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{encoded}" if with_prefix else encoded

def decode_base64_image(base64_str: str) -> Image.Image:
    # Remove the prefix if present (e.g., "data:image/png;base64,")
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image

async def async_generate_text_and_image(prompt):
    response = await client.aio.models.generate_content(
        model=os.getenv("MODEL"),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
        )
    )
    for part in response.candidates[0].content.parts:
        if hasattr(part, 'text') and part.text is not None:
            yield {'type': 'text', 'data': part.text}
        elif hasattr(part, 'inline_data') and part.inline_data is not None:
            yield {'type': 'image', 'data': bytes_to_base64(part.inline_data.data)}

async def async_generate_with_image_input(text, image_path):
    # Validate that the image input is a base64 data URI
    if not isinstance(image_path, str) or not image_path.startswith("data:image/"):
        raise ValueError("Invalid image input: expected a base64 Data URI starting with 'data:image/'")
    # Decode the base64 string into a PIL Image
    image = decode_base64_image(image_path)
    response = await client.aio.models.generate_content(
        model=os.getenv("MODEL"),
        contents=[text, image],
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
        )
    )
    for part in response.candidates[0].content.parts:
        if hasattr(part, 'text') and part.text is not None:
            yield {'type': 'text', 'data': part.text}
        elif hasattr(part, 'inline_data') and part.inline_data is not None:
            yield {'type': 'image', 'data': bytes_to_base64(part.inline_data.data)}