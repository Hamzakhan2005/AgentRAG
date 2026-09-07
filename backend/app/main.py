import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import chat, upload
from app.services.vectorstore import clear_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs ONCE per server start (not per request) - cheap, no lag.
    # Gives every new run a clean vectorstore, matching "restart = fresh session".
    logger.info("Server starting - clearing vectorstore for a fresh session")
    clear_vectorstore()
    yield
    logger.info("Server shutting down")


app = FastAPI(title="Agentic RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api")
app.include_router(chat.router, prefix="/api")

@app.get("/")
def health():
    logger.info("Health check called")
    return {"status": "running"}