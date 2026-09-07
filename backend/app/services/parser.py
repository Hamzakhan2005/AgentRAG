import logging
import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def _format_table(table: list) -> str:
    """Turn a pdfplumber table (list of rows) into readable 'Field: Value' /
    pipe-separated text so form fields and table data survive into chunks."""
    lines = []
    for row in table:
        cells = [str(cell).strip() if cell is not None else "" for cell in row]
        cells = [c for c in cells if c]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _ocr_page(file_path: str, page_number: int) -> str:
    """OCR fallback for pages where normal text extraction returns
    little/nothing (scanned pages, image-only content).

    Uses PyMuPDF (pymupdf) to render the page to an image - pip-installable,
    no ImageMagick/Ghostscript needed - then pytesseract to OCR it.
    Requires the Tesseract OCR engine installed on the system separately
    (not pip-installable). Degrades gracefully if either is missing.
    """
    try:
        import fitz  # PyMuPDF
        import pytesseract

        doc = fitz.open(file_path)
        page = doc[page_number]
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        doc.close()

        from PIL import Image
        import io
        img = Image.open(io.BytesIO(img_bytes))
        ocr_text = pytesseract.image_to_string(img)
        return ocr_text
    except ImportError as e:
        logger.warning(
            f"OCR fallback skipped ({e}). To enable OCR for scanned PDFs: "
            "1) pip install pymupdf pytesseract  "
            "2) install the Tesseract OCR engine on your system "
            "(https://github.com/tesseract-ocr/tesseract) and ensure it's on PATH."
        )
        return ""
    except Exception as e:
        logger.warning(f"OCR fallback failed for page {page_number + 1}: {e}")
        return ""


def parse_pdf(file_path: str) -> str:
    logger.info(f"Parsing PDF: {file_path}")
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""

                tables = page.extract_tables()
                table_text = ""
                if tables:
                    formatted = [_format_table(t) for t in tables if t]
                    table_text = "\n\n".join(t for t in formatted if t)
                    if table_text:
                        logger.info(f"Page {i+1}: extracted {len(tables)} table(s)")

                combined = "\n".join(part for part in [page_text, table_text] if part)

                # If a page yields almost nothing (likely scanned/image-only),
                # fall back to OCR so content isn't silently dropped.
                if len(combined.strip()) < 20:
                    logger.warning(
                        f"Page {i+1}/{len(pdf.pages)} yielded almost no text "
                        f"({len(combined.strip())} chars) - attempting OCR fallback"
                    )
                    ocr_text = _ocr_page(file_path, i)
                    if ocr_text.strip():
                        combined = ocr_text
                        logger.info(f"Page {i+1}: OCR recovered {len(ocr_text)} chars")

                text_parts.append(combined)
                logger.debug(f"Parsed page {i+1}/{len(pdf.pages)}, {len(combined)} chars")

        text = "\n".join(text_parts)
        logger.info(f"PDF parsed successfully, total chars: {len(text)}")
        return text
    except Exception as e:
        logger.error(f"Failed to parse PDF: {e}")
        raise

def parse_txt(file_path: str) -> str:
    logger.info(f"Parsing TXT: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(f"TXT parsed successfully, total chars: {len(text)}")
        return text
    except Exception as e:
        logger.error(f"Failed to parse TXT: {e}")
        raise

def parse_docx(file_path: str) -> str:
    logger.info(f"Parsing DOCX: {file_path}")
    try:
        from docx import Document as DocxDocument
        doc = DocxDocument(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        logger.info(f"DOCX parsed successfully, total chars: {len(text)}")
        return text
    except Exception as e:
        logger.error(f"Failed to parse DOCX: {e}")
        raise

def parse_file(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    logger.info(f"Detected file extension: {ext}")
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".txt":
        return parse_txt(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    else:
        logger.error(f"Unsupported file type: {ext}")
        raise ValueError(f"Unsupported file type: {ext}. Supported: pdf, txt, docx")

def chunk_text(text: str, filename: str) -> list[Document]:
    logger.info(f"Chunking text from file: {filename}")
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.create_documents(
            texts=[text],
            metadatas=[{"source": filename}]
        )
        logger.info(f"Created {len(chunks)} chunks from {filename}")
        return chunks
    except Exception as e:
        logger.error(f"Failed to chunk text: {e}")
        raise