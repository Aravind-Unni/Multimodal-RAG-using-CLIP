import os
import base64
from typing import List

from dotenv import load_dotenv
import torch
from transformers import CLIPProcessor, CLIPModel

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_ollama import ChatOllama

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "multimodal_rag"
TOP_K = 3
TEXT_CANDIDATES = 8
IMAGE_CANDIDATES = 2

# ---- CLIP model — must match the one used at storage time ----
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
clip_model.eval()


def embed_text(text: str) -> List[float]:
    inputs = clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().tolist()


client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def retrieve_multimodal(question: str, text_k: int = TEXT_CANDIDATES, image_k: int = IMAGE_CANDIDATES) -> List[Document]:
    """Search text and image points separately, so images (which score lower
    under CLIP's cross-modal similarity) aren't drowned out by text-text matches."""
    query_vector = embed_text(question)

    text_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="dense",
        limit=text_k,
        query_filter=Filter(
            must=[FieldCondition(key="metadata.content_type", match=MatchValue(value="text"))]
        ),
        with_payload=True,
    )

    image_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="dense",
        limit=image_k,
        query_filter=Filter(
            must=[FieldCondition(key="metadata.content_type", match=MatchValue(value="image"))]
        ),
        with_payload=True,
    )

    docs = []
    for point in list(text_results.points) + list(image_results.points):
        payload = point.payload
        docs.append(
            Document(
                page_content=payload.get("text", ""),
                metadata=payload.get("metadata", {}),
            )
        )
    return docs


reranker = NVIDIARerank(
    model="nvidia/llama-nemotron-rerank-vl-1b-v2",
    api_key=NVIDIA_API_KEY,
    top_n=TOP_K,
)

llm = ChatOllama(
    model="qwen2.5vl:latest",
    temperature=0,
)

SYSTEM_PROMPT = """Answer ONLY using the context and any attached images below. No outside knowledge.
If the context and images don't contain the answer, say: "I don't have enough information in the document to answer that."
If the question is unrelated to the document, say: "That question is outside the scope of this document."""


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_multimodal_message(question: str, docs: List[Document]) -> HumanMessage:
    """Text chunks go in as plain text. Image docs get their real pixels attached
    (base64), not just a path string, so a vision-capable model can actually read them."""
    text_parts = []
    image_blocks = []

    for doc in docs:
        content_type = doc.metadata.get("content_type")
        if content_type == "image":
            image_path = doc.metadata.get("image_path")
            page_number = doc.metadata.get("page_number")
            text_parts.append(f"[Chart/image on page {page_number} is attached below]")
            image_blocks.append({
                "type": "image_url",
                "image_url": f"data:image/png;base64,{encode_image(image_path)}",
            })
        else:
            text_parts.append(doc.page_content)

    context_text = "\n\n".join(text_parts)

    content = [{"type": "text", "text": f"Context:\n{context_text}\n\nQuestion: {question}"}]
    content.extend(image_blocks)

    return HumanMessage(content=content)


def answer_with_sources(question: str):
    candidates = retrieve_multimodal(question)

    text_candidates = [d for d in candidates if d.metadata.get("content_type") == "text"]
    image_candidates = [d for d in candidates if d.metadata.get("content_type") == "image"]

    reranked_text = reranker.compress_documents(query=question, documents=text_candidates)

    # Images bypass the text reranker (it can't score them) — include them as-is,
    # since there are only a couple in the whole collection anyway.
    docs = list(reranked_text) + image_candidates

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        build_multimodal_message(question, docs),
    ]

    result = llm.invoke(messages)
    return result.content, docs


def answer(question: str) -> str:
    result, _ = answer_with_sources(question)
    return result


if __name__ == "__main__":
    query = "give the Q3 revenue comparison of FY2024 vs. FY2025"
    print(answer(query))