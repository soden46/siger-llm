# inference/api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import asyncio
import json

from .generator import Generator
from .chat      import ChatSession


app = FastAPI(title="LLM API", version="1.0.0")

# Generator instance (diinit saat startup)
_generator: Optional[Generator] = None
_sessions: dict[str, ChatSession] = {}


# ── Request/Response Models ────────────────────────────────
class GenerateRequest(BaseModel):
    prompt:             str
    max_new_tokens:     int   = Field(200, ge=1, le=2048)
    temperature:        float = Field(0.8, ge=0.0, le=2.0)
    top_k:              int   = Field(50, ge=0)
    top_p:              float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.15, ge=1.0)
    lang:               Optional[str] = None
    stream:             bool  = False

class ChatRequest(BaseModel):
    session_id: str
    message:    str
    stream:     bool  = False
    temperature: float = 0.8
    max_new_tokens: int = 300

class MemoryDocumentRequest(BaseModel):
    text: str
    metadata: dict = Field(default_factory=dict)

class MemoryFactRequest(BaseModel):
    fact: str

class GenerateResponse(BaseModel):
    text:        str
    token_count: int


# ── Endpoints ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _generator is not None}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if _generator is None:
        raise HTTPException(503, "Model not loaded")

    if req.stream:
        # Streaming response (SSE-style)
        async def token_stream():
            for token in _generator.stream(
                req.prompt,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                lang=req.lang,
            ):
                data = json.dumps({"token": token})
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")

    # Normal response
    text = _generator.generate(
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        lang=req.lang,
    )
    token_count = _generator.tokenizer.count_tokens(text)
    return GenerateResponse(text=text, token_count=token_count)


@app.post("/chat")
async def chat(req: ChatRequest):
    if _generator is None:
        raise HTTPException(503, "Model not loaded")

    # Get or create session
    if req.session_id not in _sessions:
        _sessions[req.session_id] = ChatSession(_generator)

    session  = _sessions[req.session_id]
    response = session.chat(
        req.message,
        stream=req.stream,
        temperature=req.temperature,
        max_new_tokens=req.max_new_tokens,
    )
    return {"session_id": req.session_id, "response": response}


@app.delete("/chat/{session_id}")
def reset_chat(session_id: str):
    if session_id in _sessions:
        _sessions[session_id].reset()
    return {"status": "reset"}


@app.post("/chat/{session_id}/memory/document")
def add_memory_document(session_id: str, req: MemoryDocumentRequest):
    if _generator is None:
        raise HTTPException(503, "Model not loaded")

    if session_id not in _sessions:
        _sessions[session_id] = ChatSession(_generator)

    _sessions[session_id].add_document(req.text, metadata=req.metadata)
    return {"status": "ok", "memory": _sessions[session_id].memory_stats()}


@app.post("/chat/{session_id}/memory/fact")
def add_memory_fact(session_id: str, req: MemoryFactRequest):
    if _generator is None:
        raise HTTPException(503, "Model not loaded")

    if session_id not in _sessions:
        _sessions[session_id] = ChatSession(_generator)

    _sessions[session_id].add_pinned_fact(req.fact)
    return {"status": "ok", "memory": _sessions[session_id].memory_stats()}


@app.get("/chat/{session_id}/memory")
def get_memory_stats(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found")
    return _sessions[session_id].memory_stats()


# ── Startup ────────────────────────────────────────────────
def init_api(generator: Generator):
    """Panggil ini sebelum jalanin uvicorn."""
    global _generator
    _generator = generator
