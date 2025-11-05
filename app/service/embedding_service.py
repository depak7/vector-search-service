# embedding_service.py
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
import os
from google import generativeai as genai
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# 1️⃣ Configure Gemini once globally
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ Missing GOOGLE_API_KEY in .env")
genai.configure(api_key=GOOGLE_API_KEY)

# 2️⃣ Load SigLIP / CLIP image model globally once
# (SigLIP may fail under SentenceTransformer, so CLIP is stable)
image_model = SentenceTransformer("clip-ViT-B-32")
print("✅ Loaded image embedding model: clip-ViT-B-32")

# ======================================
#  Image Embedding
# ======================================
def fetch_image(image_url: str) -> Image.Image:
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise ValueError(f"❌ Failed to fetch image from URL: {e}")

def get_image_embedding(pil_image: Image.Image):
    try:
        embedding = image_model.encode(pil_image, normalize_embeddings=True)
        print(f"✅ Generated image embedding of length {len(embedding)}")
        return embedding.tolist()
    except Exception as e:
        raise RuntimeError(f"❌ Failed to generate image embedding: {e}")

# ======================================
#  Text Embedding (Gemini)
# ======================================
def get_text_embedding(text: str):
    if not text or not text.strip():
        raise ValueError("❌ Text input cannot be empty for embedding")

    try:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        embedding = response["embedding"]
        print(f"✅ Generated text embedding of length {len(embedding)}")
        return embedding
    except Exception as e:
        raise RuntimeError(f"❌ Failed to get text embedding from Gemini: {e}")
