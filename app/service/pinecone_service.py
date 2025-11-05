import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Index names
IMAGE_INDEX_NAME = os.getenv("PINECONE_IMAGE_INDEX", "ecom-fort-image-index")
TEXT_INDEX_NAME = os.getenv("PINECONE_TEXT_INDEX", "ecom-fort-text-index")

# -------------------------------------------------------------
# 🧠 Create indexes if they don't exist
# -------------------------------------------------------------
def ensure_indexes_exist():
    existing_indexes = [i["name"] for i in pc.list_indexes()]

    if IMAGE_INDEX_NAME not in existing_indexes:
        print(f"⚙️ Creating image index: {IMAGE_INDEX_NAME}")
        pc.create_index(
            name=IMAGE_INDEX_NAME,
            dimension=512,  # SigLIP image embedding size
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    if TEXT_INDEX_NAME not in existing_indexes:
        print(f"⚙️ Creating text index: {TEXT_INDEX_NAME}")
        pc.create_index(
            name=TEXT_INDEX_NAME,
            dimension=768,  # Gemini text embedding size
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    print("✅ Indexes ready.")

# Ensure both exist before usage
ensure_indexes_exist()

# Connect to them
IMAGE_INDEX = pc.Index(IMAGE_INDEX_NAME)
TEXT_INDEX = pc.Index(TEXT_INDEX_NAME)

# -------------------------------------------------------------
# 🚀 Core Functions
# -------------------------------------------------------------
def upsert_embedding(product_id: str, embedding: list, metadata: dict, index_type="image"):
    """
    Store a product embedding (image or text) in Pinecone.
    """
    index = IMAGE_INDEX if index_type == "image" else TEXT_INDEX
    vector = {
        "id": f"{index_type}-{product_id}",
        "values": embedding,
        "metadata": metadata,
    }

    index.upsert(vectors=[vector])
    print(f"✅ Upserted {index_type} embedding for product {product_id}")


def clear_index(index_type="image"):
    """
    Clears all vectors in the specified index.
    """
    index = IMAGE_INDEX if index_type == "image" else TEXT_INDEX
    index.delete(delete_all=True)
    print(f"🧹 Cleared all vectors in {index_type} index.")


def query_similar_products(
    query_embedding: list,
    top_k: int = 10,
    filters:dict ={},
    index_type: str = "image"
):

    index = IMAGE_INDEX if index_type == "image" else TEXT_INDEX
    expected_dim = 512 if index_type == "image" else 768

    if len(query_embedding) != expected_dim:
        raise ValueError(
            f"❌ Embedding dimension {len(query_embedding)} does not match {expected_dim} for {index_type} index"
        )

    try:

        response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            filter=filters or {},
        )
        

        matches = response.get("matches", [])
        results = []

        for match in matches:
            score = float(match["score"])
            normalized_score = round(score * 100, 2)  # 0–100 scale

            results.append({
                "product_id": match["id"],
                "score": normalized_score,
                "metadata": match.get("metadata", {}),
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        print(f"✅ Found {len(results)} similar products ({index_type})")

        return results

    except Exception as e:
        print(f"❌ Pinecone query failed for {index_type}: {e}")
        return []

