from fastapi import APIRouter
from ..controllers import AgenticController

from ..interfaces import MessagePayload

router = APIRouter(prefix="/chat-ai")

agenticController = AgenticController()


@router.post("/")
async def get_users(payload: MessagePayload):
    """Dùng LLM + Tools (email, list expert) — không qua RAG"""
    return await agenticController.pickToolsFunc(payload.message)


@router.post("/rag")
async def chat_with_rag(payload: MessagePayload):
    """Embedding → Qdrant search → LLM answer (RAG pipeline)"""
    return await agenticController.callAiFromClient(payload.message)