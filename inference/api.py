from __future__ import annotations

import json
import os
from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from data_governance import LearningDataIntake, LearningIntakeRecord, PromptInjectionFilter

from .generator import Generator
from .chat import ChatSession, clean_assistant_response
from .security import APIProtectionMiddleware, APISecuritySettings
from .user_feedback_collector import UserFeedbackCollector


app = FastAPI(
    title="SigerLM API",
    version="1.1.0",
    description="REST and SSE API for integrating SigerLM with web and mobile apps.",
)

_cors_origins = [
    origin.strip()
    for origin in os.environ.get("SIGER_CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=("*" not in _cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)
_security_settings = APISecuritySettings.from_env()
app.add_middleware(APIProtectionMiddleware, settings=_security_settings)

# Generator instance (diinit saat startup)
_generator: Optional[Generator] = None
_sessions: dict[str, ChatSession] = {}
_checkpoint_path: str | None = None
_learning_intake = LearningDataIntake()
_prompt_injection_filter = PromptInjectionFilter(block_high=True)


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=16000)
    max_new_tokens: int = Field(200, ge=1, le=1024)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.15, ge=1.0)
    lang: Optional[str] = None
    stream: bool = False

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(..., min_length=1, max_length=16000)
    stream: bool = False
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.15, ge=1.0)
    max_new_tokens: int = Field(300, ge=1, le=1024)

class MemoryDocumentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000)
    metadata: dict = Field(default_factory=dict)

class ToolResultRequest(BaseModel):
    output: str = Field(..., min_length=1, max_length=200000)
    command: str = Field("", max_length=1000)
    metadata: dict = Field(default_factory=dict)

class MemoryFactRequest(BaseModel):
    fact: str = Field(..., min_length=1, max_length=2000)

class FeedbackRatingRequest(BaseModel):
    user_id: str = Field("", max_length=200)
    session_id: str | None = None
    prompt: str = Field(..., min_length=1, max_length=16000)
    response: str = Field(..., min_length=1, max_length=16000)
    rating: int = Field(..., ge=1, le=5)
    model_version: str = Field("", max_length=200)
    category: str = Field("general", max_length=80)

class FeedbackPreferenceRequest(BaseModel):
    user_id: str = Field("", max_length=200)
    session_id: str | None = None
    prompt: str = Field(..., min_length=1, max_length=16000)
    chosen_response: str = Field(..., min_length=1, max_length=16000)
    rejected_response: str = Field(..., min_length=1, max_length=16000)
    rating: int = Field(5, ge=1, le=5)
    model_version: str = Field("", max_length=200)
    category: str = Field("general", max_length=80)
    approved_for_training: bool = False

class FeedbackReportRequest(BaseModel):
    user_id: str = Field("", max_length=200)
    session_id: str | None = None
    prompt: str = Field(..., min_length=1, max_length=16000)
    response: str = Field(..., min_length=1, max_length=16000)
    reason: str = Field(..., min_length=1, max_length=1000)

class LearningIntakeRequest(BaseModel):
    source_type: str = Field(..., min_length=1, max_length=80, description="web, app, feedback, document, or other")
    text: str = Field("", max_length=100000)
    instruction: str = Field("", max_length=16000)
    input: str = Field("", max_length=50000)
    output: str = Field("", max_length=50000)
    source_url: str = Field("", max_length=2000)
    app_id: str = Field("", max_length=200)
    session_id: str = Field("", max_length=200)
    user_id: str = Field("", max_length=200)
    language: str = Field("", max_length=40)
    domain: str = Field("general", max_length=80)
    purpose: str = Field("model_improvement", max_length=120)
    learning_mode: str = Field("training_candidate", max_length=80)
    consent: bool = False
    allow_training: bool = False
    approved_for_training: bool = False
    metadata: dict = Field(default_factory=dict)

class LearningBatchIntakeRequest(BaseModel):
    records: list[LearningIntakeRequest] = Field(default_factory=list, max_length=100)

class LearningApprovalRequest(BaseModel):
    intake_id: str = Field(..., min_length=1, max_length=80)
    reviewer: str = Field(..., min_length=1, max_length=120)
    decision: str = Field(..., pattern="^(approve|reject)$")
    note: str = ""

class GenerateResponse(BaseModel):
    text: str
    token_count: int

class ChatResponse(BaseModel):
    session_id: str
    response: str
    token_count: int
    memory: dict

class SessionResponse(BaseModel):
    session_id: str
    memory: dict


def to_learning_record(req: LearningIntakeRequest) -> LearningIntakeRecord:
    return LearningIntakeRecord(
        source_type=req.source_type,
        text=req.text,
        instruction=req.instruction,
        input=req.input,
        output=req.output,
        source_url=req.source_url,
        app_id=req.app_id,
        session_id=req.session_id,
        user_id=req.user_id,
        language=req.language,
        domain=req.domain,
        purpose=req.purpose,
        learning_mode=req.learning_mode,
        consent=req.consent,
        allow_training=req.allow_training,
        approved_for_training=req.approved_for_training,
        metadata=req.metadata,
    )


def require_api_key(x_siger_api_key: Optional[str] = Header(default=None)) -> None:
    configured_key = os.environ.get("SIGER_API_KEY")
    if _security_settings.require_api_key and not configured_key:
        raise HTTPException(status_code=503, detail="SIGER_API_KEY is required but not configured")
    if configured_key and x_siger_api_key != configured_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def require_model() -> Generator:
    if _generator is None:
        raise HTTPException(503, "Model not loaded")
    return _generator


def get_or_create_session(session_id: str | None) -> tuple[str, ChatSession]:
    if _generator is None:
        raise HTTPException(503, "Model not loaded")
    resolved_id = session_id or str(uuid4())
    if resolved_id not in _sessions:
        _sessions[resolved_id] = ChatSession(_generator)
    return resolved_id, _sessions[resolved_id]


def sse(data: dict | str, event: str | None = None) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {payload}\n\n"


def reject_prompt_injection(text: str, context: str = "chat") -> dict:
    scan = _prompt_injection_filter.scan_text(text, context=context)
    if not scan.allowed:
        raise HTTPException(
            status_code=400,
            detail={
                "message": _prompt_injection_filter.safe_response(),
                "injection_scan": scan.to_dict(),
            },
        )
    return scan.to_dict()


def mark_untrusted_metadata(metadata: dict, *texts: str) -> dict:
    combined = "\n".join(text for text in texts if text)
    scan = _prompt_injection_filter.scan_text(combined, context="external_context")
    if scan.findings:
        return {
            **metadata,
            "untrusted_external_content": True,
            "prompt_injection_scan": scan.to_dict(),
        }
    return metadata


# Endpoints
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _generator is not None,
        "sessions": len(_sessions),
    }


@app.get("/v1/status")
def status():
    return {
        "status": "ok",
        "model_loaded": _generator is not None,
        "checkpoint": _checkpoint_path,
        "sessions": len(_sessions),
        "auth_required": bool(os.environ.get("SIGER_API_KEY")),
        "cors_origins": os.environ.get("SIGER_CORS_ORIGINS", "*"),
        "security": {
            "require_api_key": _security_settings.require_api_key,
            "max_body_bytes": _security_settings.max_body_bytes,
            "rate_limit_requests": _security_settings.rate_limit_requests,
            "rate_limit_window_seconds": _security_settings.rate_limit_window_seconds,
        },
    }


@app.post("/generate", response_model=GenerateResponse)
@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(
    req: GenerateRequest,
    _auth: None = Depends(require_api_key),
    generator: Generator = Depends(require_model),
):
    reject_prompt_injection(req.prompt, context="chat")
    if req.stream:
        # Streaming response (SSE-style)
        async def token_stream():
            for token in generator.stream(
                req.prompt,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                lang=req.lang,
            ):
                yield sse({"token": token})
            yield sse("[DONE]")

        return StreamingResponse(token_stream(), media_type="text/event-stream")

    # Normal response
    text = generator.generate(
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        lang=req.lang,
    )
    token_count = generator.tokenizer.count_tokens(text)
    return GenerateResponse(text=text, token_count=token_count)


@app.post("/chat", response_model=ChatResponse)
@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    _auth: None = Depends(require_api_key),
):
    session_id, session = get_or_create_session(req.session_id)
    reject_prompt_injection(req.message, context="chat")

    if req.stream:
        async def chat_token_stream():
            effective_user_input = session.memory.ingest_long_user_message(
                req.message,
                max_inline_chars=session.long_input_threshold_chars,
                chunk_size_words=session.document_chunk_size_words,
                overlap_words=session.document_overlap_words,
            )
            prompt = session._build_prompt(effective_user_input)
            full_response = ""
            yield sse({"session_id": session_id}, event="start")

            for token in session.generator.stream(
                prompt,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
            ):
                full_response += token
                yield sse({"token": token})

            response = clean_assistant_response(full_response)
            session.history.append({"role": "user", "content": effective_user_input})
            session.history.append({"role": "assistant", "content": response})
            if len(session.history) > session.max_history * 2:
                session.history = session.history[-session.max_history * 2 :]
            session.memory.add_turn("user", effective_user_input)
            session.memory.add_turn("assistant", response)
            yield sse(
                {
                    "session_id": session_id,
                    "response": response,
                    "memory": session.memory_stats(),
                },
                event="done",
            )
            yield sse("[DONE]")

        return StreamingResponse(chat_token_stream(), media_type="text/event-stream")

    response = session.chat(
        req.message,
        stream=False,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        max_new_tokens=req.max_new_tokens,
    )
    return ChatResponse(
        session_id=session_id,
        response=response,
        token_count=session.generator.tokenizer.count_tokens(response),
        memory=session.memory_stats(),
    )


@app.post("/v1/sessions", response_model=SessionResponse)
def create_session(_auth: None = Depends(require_api_key)):
    session_id, session = get_or_create_session(None)
    return SessionResponse(session_id=session_id, memory=session.memory_stats())


@app.delete("/chat/{session_id}")
@app.delete("/v1/chat/{session_id}")
def reset_chat(session_id: str, _auth: None = Depends(require_api_key)):
    if session_id in _sessions:
        _sessions[session_id].reset()
    return {"status": "reset"}


@app.post("/chat/{session_id}/memory/document")
@app.post("/v1/chat/{session_id}/memory/document")
def add_memory_document(
    session_id: str,
    req: MemoryDocumentRequest,
    _auth: None = Depends(require_api_key),
):
    _, session = get_or_create_session(session_id)
    metadata = mark_untrusted_metadata(req.metadata, req.text)
    session.add_document(req.text, metadata=metadata)
    return {"status": "ok", "session_id": session_id, "memory": session.memory_stats()}


@app.post("/chat/{session_id}/memory/tool-result")
@app.post("/v1/chat/{session_id}/memory/tool-result")
def add_tool_result(
    session_id: str,
    req: ToolResultRequest,
    _auth: None = Depends(require_api_key),
):
    _, session = get_or_create_session(session_id)
    metadata = mark_untrusted_metadata(req.metadata, req.output, req.command)
    result = session.add_tool_result(
        req.output,
        command=req.command,
        metadata=metadata,
    )
    return {
        "status": "ok",
        "session_id": session_id,
        "compression": result.metadata(),
        "memory": session.memory_stats(),
    }


@app.post("/chat/{session_id}/memory/fact")
@app.post("/v1/chat/{session_id}/memory/fact")
def add_memory_fact(
    session_id: str,
    req: MemoryFactRequest,
    _auth: None = Depends(require_api_key),
):
    _, session = get_or_create_session(session_id)
    metadata = reject_prompt_injection(req.fact, context="external_context")
    if metadata.get("findings"):
        raise HTTPException(status_code=400, detail={"message": "Pinned fact contains prompt-injection patterns.", "injection_scan": metadata})
    session.add_pinned_fact(req.fact)
    return {"status": "ok", "session_id": session_id, "memory": session.memory_stats()}


@app.get("/chat/{session_id}/memory")
@app.get("/v1/chat/{session_id}/memory")
def get_memory_stats(session_id: str, _auth: None = Depends(require_api_key)):
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found")
    return _sessions[session_id].memory_stats()


@app.post("/v1/feedback/rating")
def submit_feedback_rating(
    req: FeedbackRatingRequest,
    _auth: None = Depends(require_api_key),
):
    reject_prompt_injection(req.prompt, context="external_context")
    collector = UserFeedbackCollector()
    feedback_id = collector.submit_quick_rating(
        user_id=req.user_id or req.session_id or "anonymous",
        prompt=req.prompt,
        response=req.response,
        rating=req.rating,
        model_version=req.model_version,
        category=req.category,
    )
    return {"status": "ok", "feedback_id": feedback_id}


@app.post("/v1/feedback/preference")
def submit_feedback_preference(
    req: FeedbackPreferenceRequest,
    _auth: None = Depends(require_api_key),
):
    reject_prompt_injection(f"{req.prompt}\n{req.chosen_response}\n{req.rejected_response}", context="external_context")
    collector = UserFeedbackCollector()
    feedback_id = collector.submit_preference(
        user_id=req.user_id or req.session_id or "anonymous",
        prompt=req.prompt,
        chosen_response=req.chosen_response,
        rejected_response=req.rejected_response,
        rating=req.rating,
        model_version=req.model_version,
        category=req.category,
        approved_for_training=req.approved_for_training,
    )
    return {
        "status": "ok",
        "feedback_id": feedback_id,
        "exportable_for_dpo": req.approved_for_training,
    }


@app.post("/v1/feedback/report")
def submit_feedback_report(
    req: FeedbackReportRequest,
    _auth: None = Depends(require_api_key),
):
    reject_prompt_injection(f"{req.prompt}\n{req.response}\n{req.reason}", context="external_context")
    collector = UserFeedbackCollector()
    feedback_id = collector.flag_response(
        user_id=req.user_id or req.session_id or "anonymous",
        prompt=req.prompt,
        response=req.response,
        reason=req.reason,
    )
    return {"status": "ok", "feedback_id": feedback_id}


@app.post("/v1/learning/intake")
def submit_learning_intake(
    req: LearningIntakeRequest,
    _auth: None = Depends(require_api_key),
):
    req.metadata.update(
        mark_untrusted_metadata(
            req.metadata,
            req.text,
            req.instruction,
            req.input,
            req.output,
        )
    )
    result = _learning_intake.submit(to_learning_record(req))
    return result


@app.post("/v1/learning/intake/batch")
def submit_learning_intake_batch(
    req: LearningBatchIntakeRequest,
    _auth: None = Depends(require_api_key),
):
    results = [_learning_intake.submit(to_learning_record(item)) for item in req.records]
    return {
        "status": "ok",
        "count": len(results),
        "results": results,
    }


@app.post("/v1/learning/approval")
def review_learning_intake(
    req: LearningApprovalRequest,
    _auth: None = Depends(require_api_key),
):
    try:
        return _learning_intake.approve(
            intake_id=req.intake_id,
            reviewer=req.reviewer,
            decision=req.decision,
            note=req.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/learning/stats")
def learning_intake_stats(_auth: None = Depends(require_api_key)):
    return _learning_intake.stats()


# Startup
def init_api(generator: Generator, checkpoint_path: str | None = None):
    """Panggil ini sebelum jalanin uvicorn."""
    global _generator, _checkpoint_path
    _generator = generator
    _checkpoint_path = checkpoint_path
