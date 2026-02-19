"""
Ingredient Health Analyzer Agent
Powered by Google Gemini Vision + LangChain

Setup:
    pip install -r requirements.txt
    Set GOOGLE_API_KEY in your .env file
"""

import os
import base64
import json
import re
import io

from dotenv import load_dotenv
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(title="Ingredient Health Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2,
        convert_system_message_to_human=True,
    )

# ──────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert nutritionist and food scientist AI.
Analyze the product ingredient list in the image.

Return ONLY a valid JSON object (no markdown, no extra text):
{
  "product_detected": "product name if visible",
  "ingredients_found": ["list", "of", "ingredients"],
  "health_score": <integer 0-100>,
  "verdict": "EXCELLENT | GOOD | MODERATE | POOR | VERY POOR",
  "summary": "2-3 sentence overview",
  "positive_aspects": ["healthy ingredients or attributes"],
  "concerns": ["concerning ingredients"],
  "harmful_ingredients": ["dangerous additives if any"],
  "recommended_for": ["suitable groups"],
  "avoid_if": ["groups who should avoid"],
  "healthier_alternatives": ["2-3 better options"],
  "detailed_analysis": "in-depth paragraph"
}

Health Score Guide:
  90-100 -> EXCELLENT  |  70-89 -> GOOD  |  50-69 -> MODERATE
  30-49  -> POOR       |  0-29  -> VERY POOR
"""

# ──────────────────────────────────────────────
# Image processing
# ──────────────────────────────────────────────
def process_image(image_bytes: bytes, mime_type: str) -> tuple:
    """Resize image and return (base64, mime_type)."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail((1600, 1600), Image.LANCZOS)

    buf = io.BytesIO()
    fmt = "JPEG" if "jpeg" in mime_type or "jpg" in mime_type else "PNG"
    img.save(buf, format=fmt, quality=88)

    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64, mime_type

# ──────────────────────────────────────────────
# Core analysis
# ──────────────────────────────────────────────
def analyze(image_bytes: bytes, mime_type: str) -> dict:
    """Send image to Gemini and return parsed JSON result."""
    b64, mime = process_image(image_bytes, mime_type)
    llm = get_llm()

    message = HumanMessage(content=[
        {
            "type": "text",
            "text": (
                "Analyze the ingredient list in this image. "
                "Return ONLY a valid JSON object."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        },
    ])

    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), message])

    raw = response.content.strip()
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("Could not parse Gemini response as JSON")

# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "online", "service": "Ingredient Health Analyzer"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_product(image: UploadFile = File(...)):
    """
    Upload a product ingredient image.
    Returns full health analysis with score 0-100.
    """
    allowed = {"image/jpeg", "image/png", "image/webp"}
    if image.content_type not in allowed:
        raise HTTPException(400, f"Unsupported file type: {image.content_type}")

    image_bytes = await image.read()
    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(400, "Image too large. Max 5MB.")

    try:
        result = analyze(image_bytes, image.content_type)
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")

    return result
