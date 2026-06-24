from fastapi import FastAPI
from app.api.summarize import router

app = FastAPI(
    title="RAG Summarizer"
)

app.include_router(router)