from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Optional
from backend.utils import async_generate_text_and_image, async_generate_with_image_input
import backend.config as config  # keep for reference if needed
import traceback

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextGenerateRequest(BaseModel):
    prompt: str

class ImageTextGenerateRequest(BaseModel):
    text: Optional[str] = None
    image: str

class Part(BaseModel):
    type: Literal["text", "image"]
    data: str

class GenerationResponse(BaseModel):
    results: List[Part]

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: TextGenerateRequest):
    """
    Generate text and image from a text prompt.
    """
    try:
        results = []
        # print(request)
        async for part in async_generate_text_and_image(request.prompt):
            results.append(part)
        return GenerationResponse(results=results)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.post("/generate_with_image", response_model=GenerationResponse)
async def generate_with_image(request: ImageTextGenerateRequest):
    """
    Generate text and image given a text and base64 image.
    """
    try:
        # print(config.PATENT_INSTRUCTION)
        # text = request.text if request.text is not None else config.PATENT_INSTRUCTION
        text = config.PATENT_INSTRUCTION
        print(text)
        results = []
        async for part in async_generate_with_image_input(text, request.image):
            results.append(part)
        return GenerationResponse(results=results)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.get("/")
async def read_root():
    return {"message": "Image generation API is up"}