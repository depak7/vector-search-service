from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.service.embedding_service import (
    fetch_image,
    get_image_embedding,
    get_text_embedding,
)
from app.service.pinecone_service import upsert_embedding, query_similar_products
from io import BytesIO
from PIL import Image


router = APIRouter()

router = APIRouter()


# ✅ Define product schema
class ProductRequest(BaseModel):
    imageUrl: str
    name: str
    description: str
    category: str
    brand: Optional[str] = None
    price: Optional[float] = None
    productId: Optional[str] = None


@router.post("/embed-products")
def embed_multiple_products(req: List[ProductRequest]):
    """
    Generate and store image + text embeddings for multiple products.
    Returns a detailed summary for success and failures.
    """
    success, failed = [], []

    for product in req:
        try:
            # Step 1️⃣ Fetch and embed image
            pil_image = fetch_image(product.imageUrl)
            image_emb = get_image_embedding(pil_image)

            # Step 2️⃣ Generate text embedding
            text_input = (
                f"{product.name}. {product.description}. "
                f"Category: {product.category}. Brand: {product.brand or ''}. Price: {product.price or ''}."
            )
            text_emb = get_text_embedding(text_input)

            # Step 3️⃣ Prepare metadata
            product_id = product.productId or product.name.lower().replace(" ", "_")
            metadata = {
                "product_id": product_id,
                "name": product.name,
                "description": product.description,
                "category": product.category
            }

            # Step 4️⃣ Store image embedding
            upsert_embedding(
                product_id=f"{product_id}_img",
                embedding=image_emb,
                metadata={**metadata, "type": "image"},
                index_type="image",
            )

            # Step 5️⃣ Store text embedding
            upsert_embedding(
                product_id=f"{product_id}_txt",
                embedding=text_emb,
                metadata={**metadata, "type": "text"},
                index_type="text",
            )

            success.append(
                {
                    "product_id": product_id,
                    "status": "success",
                    "stored_types": ["image", "text"],
                }
            )

        except Exception as e:
            failed.append(
                {
                    "product_name": product.name,
                    "error": str(e),
                }
            )

    # Step 6️⃣ Build final response
    return {
        "status": True if success else False,
        "summary": {
            "processed": len(req),
            "success_count": len(success),
            "failed_count": len(failed),
        },
        "success": success,
        "failed": failed,
    }


@router.post("/image-search")
async def similar_product_image_search(
    image_url: str = Form(None),
    file: UploadFile | None = File(None),
    top_k: int = Form(10),
):
    """
    Perform visual search using an uploaded image or an image URL.
    - If both are provided, `file` takes priority.
    - Returns top_k visually similar products from Pinecone.
    """
    try:
        # ✅ Step 1: Validate input
        if (not file or file.filename.strip() == "") and not image_url:
            raise HTTPException(
                status_code=400, detail="Please provide either image_url or file."
            )

        # ✅ Step 2: Read image
        if file and file.filename.strip() != "":
            try:
                image_bytes = await file.read()
                pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                source = "file"
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid uploaded image: {e}"
                )
        else:
            try:
                pil_image = fetch_image(image_url)
                source = "url"
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Failed to fetch image from URL: {e}"
                )

        # ✅ Step 3: Generate embedding
        embedding = get_image_embedding(pil_image)

        # ✅ Step 4: Query Pinecone for similar products
        results = query_similar_products(
            query_embedding=embedding, top_k=top_k, filters=None, index_type="image"
        )

        return {
            "status": True,
            "query_type": source,
            "count": len(results),
            "results": results,
        }

    except HTTPException as e:
        raise e  # Re-raise handled HTTP errors cleanly
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"❌ Failed to perform image search: {str(e)}"
        )


@router.post("/text-search")
async def similar_product_text_search(query: str = Form(...), top_k: int = Form(10)):
    """
    Perform semantic product search using a text query (e.g. 'red running shoes').
    - Uses text embeddings from Gemini.
    - Searches the Pinecone text index.
    - Optionally excludes a product_id (useful for recommendations).
    """
    try:
        # ✅ Step 1: Validate query
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        embedding = get_text_embedding(query)

        results = query_similar_products(
            query_embedding=embedding, top_k=top_k, filters=None, index_type="text"
        )

        return {
            "status": True,
            "query": query,
            "count": len(results),
            "results": results,
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to perform text search: {str(e)}"
        )


@router.post("/recommendations")
async def get_recommendations(
    product_id: str = Form(...),
    product_name: str = Form(None),
    description: str = Form(None),
    image_url: str = Form(None),
    mode: str = Form("hybrid"),
    top_k: int = Form(10),
):
    try:
        if not product_id:
            raise HTTPException(status_code=400, detail="Product ID is required.")

        # ✅ Build base metadata filter — exclude the same product
        filters = {"product_id": {"$ne": product_id}}

        combined_results = []

        # ✅ 1️⃣ IMAGE-based recommendations
        if mode in ("image", "hybrid") and image_url:
            try:
                pil_image = fetch_image(image_url)
                image_emb = get_image_embedding(pil_image)
                image_results = query_similar_products(
                    query_embedding=image_emb,
                    top_k=top_k,
                    filters={"type": {"$eq": "image"}, **filters},
                    index_type="image",
                )
                combined_results.extend(image_results)
            except Exception as e:
                print(f"⚠️ Image recommendation failed: {e}")

        # ✅ 2️⃣ TEXT-based recommendations
        if mode in ("text", "hybrid") and (product_name or description):
            try:
                text_input = f"{product_name or ''}. {description or ''}".strip()
                text_emb = get_text_embedding(text_input)
                text_results = query_similar_products(
                    query_embedding=text_emb,
                    top_k=top_k,
                    filters={"type": {"$eq": "text"}, **filters},
                    index_type="text",
                )
                combined_results.extend(text_results)
            except Exception as e:
                print(f"⚠️ Text recommendation failed: {e}")

        if not combined_results:
            raise HTTPException(status_code=404, detail="No recommendations found.")

        # ✅ 3️⃣ Merge + rank (for hybrid)
        final_results = _deduplicate_and_rank(combined_results)

        return {
            "status": True,
            "mode": mode,
            "count": len(final_results),
            "recommendations": final_results,
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"❌ Failed to generate recommendations: {str(e)}"
        )


def _deduplicate_and_rank(results):
    """
    Deduplicate products by ID and rank by score (descending).
    If same product appears from both text & image, keep the best score.
    """
    merged = {}
    for item in results:
        pid = item["product_id"]
        score = item["score"]
        if pid not in merged or score > merged[pid]["score"]:
            merged[pid] = item

    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)
