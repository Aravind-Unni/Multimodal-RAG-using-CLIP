from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

EMBED_MODEL_ID = "openai/clip-vit-base-patch32"
MAX_TOKENS = 77  # CLIP's text encoder hard limit


def chunk_document(md_path):
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
        max_tokens=MAX_TOKENS,
    )

    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
    )

    converter = DocumentConverter()
    result = converter.convert(md_path)

    chunk_iter = chunker.chunk(dl_doc=result.document)
    chunks = list(chunk_iter)

    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        enriched_text = chunker.contextualize(chunk=chunk)
        print(f"=== Chunk {i} ===")
        print(f"enriched_text:\n{enriched_text[:300]!r}…\n")

        enriched_chunks.append({
            "chunk": chunk,          # keep raw chunk for metadata (page number etc.)
            "text": enriched_text,   # what actually gets embedded + stored
        })

    return enriched_chunks


if __name__ == "__main__":
    chunk_document(r"C:\RAG_CLIP\output\output.md")