import os
from typing import List, Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hybrid_search import answer_with_sources

IMAGE_DIR = r"C:\RAG_CLIP\output\images"

app = FastAPI(title="Multimodal RAG API")

# Serve extracted images directly so the frontend can <img src="/images/xyz.png">
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")


class QuestionRequest(BaseModel):
    question: str


class SourceResponse(BaseModel):
    content_type: str
    text: Optional[str] = None
    image_url: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceResponse]


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    result_text, docs = answer_with_sources(request.question)

    sources = []
    for doc in docs:
        content_type = doc.metadata.get("content_type")

        if content_type == "image":
            image_path = doc.metadata.get("image_path")
            filename = os.path.basename(image_path)
            sources.append(SourceResponse(content_type="image", image_url=f"/images/{filename}"))
        else:
            sources.append(SourceResponse(content_type="text", text=doc.page_content))

    return AnswerResponse(answer=result_text, sources=sources)


# Serve the static frontend last, so it doesn't shadow the /ask and /images routes above
app.mount("/", StaticFiles(directory="frontend/static", html=True), name="frontend")