import os
import glob
import uuid
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
)
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv

from chunking import chunk_document

load_dotenv()

MODEL_ID = "openai/clip-vit-base-patch32"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "multimodal_rag"

clip_model = CLIPModel.from_pretrained(MODEL_ID)
clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)
clip_model.eval()

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def embed_image(image_data):
    """Embed image using CLIP"""
    if isinstance(image_data, str):  # If path
        image = Image.open(image_data).convert("RGB")
    else:  # If PIL Image
        image = image_data

    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()


def embed_text(text):
    """Embed text using CLIP."""
    inputs = clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,  # CLIP's max token length
    )
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)

        # Defensive unwrap in case get_text_features ever returns a ModelOutput
        if not torch.is_tensor(features):
            features = getattr(features, "text_embeds", None) or getattr(features, "pooler_output", None)
            if features is None:
                raise TypeError(f"Unexpected output type from get_text_features: {type(features)}")

        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()


def embed_sparse(text):
    sparse_embedding = next(sparse_model.embed([text]))
    return SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist(),
    )


def create_collection():
    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=512, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )
    print(f"Created collection: {COLLECTION_NAME}")


def store_text_chunk(chunk_text, doc_id, page_number):
    dense_vector = embed_text(chunk_text)
    sparse_vector = embed_sparse(chunk_text)

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": dense_vector.tolist(),
            "sparse": sparse_vector,
        },
        payload={
            "text": chunk_text,
            "metadata": {
                "doc_id": doc_id,
                "page_number": page_number,
                "content_type": "text",
            },
        },
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])


def store_image(image_path, doc_id, page_number):
    dense_vector = embed_image(image_path)

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": dense_vector.tolist(),
        },
        payload={
            "text": "",  # QdrantVectorStore expects the content key to exist
            "metadata": {
                "doc_id": doc_id,
                "page_number": page_number,
                "content_type": "image",
                "image_path": image_path,
            },
        },
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])


def get_chunk_page_number(chunk):
    """Best-effort extraction of the source page number from a docling chunk."""
    try:
        return chunk.meta.doc_items[0].prov[0].page_no
    except (AttributeError, IndexError):
        return None


def store_document(md_path, image_dir, doc_id):
    create_collection()

    # ---- text chunks ----
    enriched_chunks = chunk_document(md_path)
    for item in enriched_chunks:
        chunk = item["chunk"]
        enriched_text = item["text"]
        page_number = get_chunk_page_number(chunk)
        store_text_chunk(enriched_text, doc_id=doc_id, page_number=page_number)
    print(f"Stored {len(enriched_chunks)} text chunks")

    # ---- images ----
    image_paths = glob.glob(os.path.join(image_dir, f"{doc_id}_*"))
    for image_path in image_paths:
        # page number encoded in filename by extract_image.py: "{doc_id}_p{page}_img{n}.ext"
        filename = os.path.basename(image_path)
        page_number = None
        try:
            page_part = filename.split("_p")[1].split("_img")[0]
            page_number = int(page_part)
        except (IndexError, ValueError):
            pass

        store_image(image_path, doc_id=doc_id, page_number=page_number)
    print(f"Stored {len(image_paths)} images")


if __name__ == "__main__":
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")

    store_document(
        md_path=r"C:\RAG_CLIP\output\output.md",
        image_dir=r"C:\RAG_CLIP\output\images",
        doc_id="ACME_Corp_Financial_Report",
    )