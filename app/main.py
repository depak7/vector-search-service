from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.controller import search_controller

# Initialize FastAPI app
app = FastAPI(
    title="AI Product Search Service",
    description="E-commerce visual + semantic search backend powered by Pinecone & Gemini",
    version="1.0.0"
)

# Enable CORS (optional — useful for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(search_controller.router, prefix="/api/products", tags=["Products"])

@app.get("/")
def root():
    return {"status": True, "message": "🚀 AI Product Search Service is running!"}
