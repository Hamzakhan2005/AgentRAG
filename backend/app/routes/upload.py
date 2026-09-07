import logging
import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.parser import parse_file, chunk_text
from app.services.vectorstore import (
    add_documents,
    register_uploaded_document,
    session_document_limit_reached,
    session_document_count,
    MAX_DOCUMENTS_PER_SESSION,
    get_session_filenames,
)

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    logger.info(f"Received upload request: {file.filename}, size: {file.size}")

    # Check the cap FIRST, before any parsing/embedding work - fail fast,
    # don't waste CPU on a file we're going to reject anyway.
    if session_document_limit_reached():
        logger.warning(
            f"Session document limit reached ({MAX_DOCUMENTS_PER_SESSION}), rejecting upload"
        )
        raise HTTPException(
            status_code=429,
            detail=(
                f"Session limit of {MAX_DOCUMENTS_PER_SESSION} documents reached. "
                "Restart the server for a fresh session, or clear documents via "
                "the DELETE /api/documents endpoint."
            ),
        )

    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Rejected unsupported file type: {ext}")
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Use pdf, txt, or docx.")

    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        logger.info(f"Saving file to: {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("File saved successfully")

        text = parse_file(file_path)

        if not text.strip():
            logger.warning("Parsed text is empty, rejecting file")
            raise HTTPException(status_code=422, detail="Could not extract text from file. File may be empty or scanned image.")

        chunks = chunk_text(text, file.filename)
        add_documents(chunks)
        register_uploaded_document(file.filename)

        logger.info(f"Upload pipeline complete for: {file.filename}")
        return {
            "message": "File uploaded and indexed successfully",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "session_documents": session_document_count(),
            "session_limit": MAX_DOCUMENTS_PER_SESSION,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Temp file cleaned up: {file_path}")

from app.services.vectorstore import clear_vectorstore, get_document_count

@router.delete("/documents")
def clear_documents():
    logger.info("Clear documents endpoint called")
    try:
        clear_vectorstore()
        return {"message": "All documents cleared from vectorstore"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/count")
def document_count():
    count = get_document_count()
    return {
        "chunk_count": count,
        "session_documents": get_session_filenames(),
        "session_document_count": session_document_count(),
        "session_limit": MAX_DOCUMENTS_PER_SESSION,
    }