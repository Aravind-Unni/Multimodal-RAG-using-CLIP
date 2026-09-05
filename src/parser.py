import os
from docling.document_converter import DocumentConverter


def convert_pdf_to_markdown(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    markdown_text = result.document.export_to_markdown()

    output_path = os.path.join(output_dir, "output.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print(f"Saved markdown -> {output_path}")
    return output_path


if __name__ == "__main__":
    convert_pdf_to_markdown(
        pdf_path=r"C:\RAG_CLIP\data\ACME_Corp_Financial_Report.pdf",
        output_dir=r"C:\RAG_CLIP\output",
    )