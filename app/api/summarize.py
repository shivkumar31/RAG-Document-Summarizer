import os
import shutil

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException
)

from app.services.rag_service import summarize_content

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/summarize")
async def summarize(
    file: UploadFile = File(default=None),
    text: str = Form(default=None),
    prompt: str = Form(default="Provide a detailed summary.")
):
    is_file_empty = file is None or not file.filename
    is_text_empty = text is None or not text.strip()


    if is_file_empty and is_text_empty:
        raise HTTPException(
            status_code=400,
            detail="You must provide a PDF file, text content, or both."
        )

    file_path = None

    try:
        
        if not is_file_empty:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

   
        summary = summarize_content(
            file_path=file_path,
            text=text if not is_text_empty else None,
            user_prompt=prompt
        )

    finally:
        
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

    return {
        "summary": summary
    }