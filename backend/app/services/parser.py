import logging
import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def parse_pdf(file_path: str) -> str:
    logger.info(f"Parsing PDF: {file_path}")
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                logger.debug(f"Parsed page {i+1}/{len(pdf.pages)}")
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