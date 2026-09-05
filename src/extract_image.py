import os
import fitz  


def extract_images(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    doc = fitz.open(pdf_path)

    image_paths = []
    image_count = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)

            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_count += 1
            image_filename = f"{doc_id}_p{page_index + 1}_img{image_count}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_paths.append(image_path)

    doc.close()

    print(f"Extracted {image_count} images -> {output_dir}")
    return image_paths


if __name__ == "__main__":
    extract_images(
        pdf_path=r"C:\RAG_CLIP\data\ACME_Corp_Financial_Report.pdf",
        output_dir=r"C:\RAG_CLIP\output\images",
    )