"""FastAPI server for the SMARTA maintenance assistant."""

from contextlib import asynccontextmanager
from contextvars import ContextVar
import logging
from pathlib import Path
import re
import sys
import json
import math
import time
from typing import Any, Literal
from datetime import datetime
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import (
    BaseModel, ConfigDict, Field, StrictInt, StrictStr,
    field_validator, model_validator,
)

from data_store import corpus_stats, initialize_corpus, list_maintenance_records
from document_loader import mysql_credentials_available
from rag.retriever import retrieve
from rag.generator import GenerationFailure, _prediction_question_intent, generate_answer
from predictive.ai4i import ModelUnavailable, Predictor

_corpus_status: dict[str, Any] = {"documents": 0, "maintenance_records": 0, "sources": 0}
_ingestion_error: str | None = None
_predictor = Predictor()
_logger = logging.getLogger("smarta.chat")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _logger.addHandler(logging.StreamHandler())
_logger.propagate = False
_chat_timing_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "smarta_chat_timing_context", default=None
)
MODE_NAMES = {
    "TWF": "tool wear failure", "HDF": "heat dissipation failure",
    "PWF": "power failure", "OSF": "overstrain failure", "RNF": "random failure",
}
OLLAMA_MODEL = "qwen2.5:3b"


def _retrieve_prediction_troubleshooting(question: str, base_query: str):
    """Fuse a few focused lexical searches without losing original source metadata."""
    queries = (
        base_query,
        f"{question} symptoms warning signs diagnostic checks inspection",
        f"{question} causes troubleshooting maintenance procedure monitoring measurements",
    )
    scores: dict[tuple[str, str], float] = {}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for query in queries:
        for rank, item in enumerate(retrieve(query, k=6), start=1):
            key = (str(item.get("source", "")), str(item.get("locator", "")))
            if not key[0] or not key[1] or not item.get("text"):
                continue
            records.setdefault(key, item)
            # Reciprocal rank fusion is insensitive to lexical score scales.
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
    ordered = sorted(scores, key=lambda key: (-scores[key], key[0], key[1]))
    return [records[key] for key in ordered[:6]]


def _ollama_model_status() -> str:
    """Check the local Ollama API and required model without blocking startup."""
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("models", []) if isinstance(payload, dict) else []
        names = {str(model.get("name", "")) for model in models if isinstance(model, dict)}
        return "ready" if OLLAMA_MODEL in names else "model_missing"
    except Exception:
        return "unavailable"


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _corpus_status, _ingestion_error
    try:
        _corpus_status = initialize_corpus()
    except Exception:
        # Database drivers can include a credential-bearing connection URL in
        # exception messages. Keep the health state useful without returning
        # or printing that diagnostic verbatim.
        _ingestion_error = "corpus_initialization_failed"
        print("SMARTA corpus initialization failed.")
    yield


app = FastAPI(title="SMARTA API", version="2.0.0", lifespan=lifespan)


@app.middleware("http")
async def measure_chat_http_timing(request: Request, call_next):
    if request.url.path != "/chat":
        return await call_next(request)
    started = time.perf_counter()
    state: dict[str, Any] = {"request_started": started}
    token = _chat_timing_context.set(state)
    try:
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        route_ms = state.get("route_ms")
        _logger.info(
            "chat_http_timing %s",
            json.dumps(
                {
                    "http_dispatch_ms": elapsed_ms,
                    "pre_route_validation_parse_ms": state.get("pre_route_validation_parse_ms"),
                    "response_finalization_ms": (
                        round(max(0.0, elapsed_ms - route_ms), 2)
                        if isinstance(route_ms, (int, float)) else None
                    ),
                    "status_code": response.status_code,
                },
                sort_keys=True,
            ),
        )
        return response
    finally:
        _chat_timing_context.reset(token)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class _PredictionEvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class _ClassifierOutput(_PredictionEvidenceModel):
    classification: StrictInt
    score: float = Field(ge=0, le=1)
    score_type: Literal["uncalibrated_random_forest_score"]

    @field_validator("classification")
    @classmethod
    def valid_classification(cls, value):
        if value not in (0, 1):
            raise ValueError("classification must be 0 or 1")
        return value


class _TrainingRange(_PredictionEvidenceModel):
    value: float
    training_range: list[float]

    @field_validator("training_range")
    @classmethod
    def valid_range(cls, bounds):
        if len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError("training_range must contain ordered lower and upper bounds")
        return bounds


class ProductionPredictionContext(_PredictionEvidenceModel):
    """Allowlisted, validated subset of one successful production /predict result."""

    provenance: Literal["production_ai4i"]
    experimental: Literal[False]
    model_id: Literal["ai4i-rf-v1"]
    machine_id: StrictStr
    prediction_timestamp: datetime
    model_version: Literal["ai4i-rf-v1"]
    machine_failure: _ClassifierOutput
    failure_modes: dict[StrictStr, _ClassifierOutput]
    failure_mode_labels: list[StrictStr]
    unsupported_failure_modes: dict[StrictStr, StrictStr]
    input_features: dict[StrictStr, Any]
    out_of_training_range: dict[StrictStr, _TrainingRange]
    limitations: StrictStr

    @model_validator(mode="after")
    def valid_context(self):
        machine_id = self.machine_id
        timestamp = self.prediction_timestamp
        if not machine_id.strip() or len(machine_id) > 128 or any(ord(char) < 32 for char in machine_id):
            raise ValueError("machine_id must be non-empty, at most 128 characters, and contain no control characters")
        if timestamp is not None and timestamp.utcoffset() is None:
            raise ValueError("prediction_timestamp must include a timezone")
        inputs = self.input_features
        expected = {
            "Type", "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        }
        if set(inputs) != expected:
            raise ValueError("input_features must match the six AI4I model features")
        if not isinstance(inputs.get("Type"), str) or inputs["Type"] not in {"L", "M", "H"}:
            raise ValueError("input_features Type must be L, M, or H")
        for name in expected - {"Type"}:
            value = inputs[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"input feature {name} must be numeric")
        supported_modes = set(self.failure_modes)
        unsupported_modes = set(self.unsupported_failure_modes)
        if (supported_modes | unsupported_modes) - {"TWF", "HDF", "PWF", "OSF", "RNF"}:
            raise ValueError("failure mode metadata contains an unknown mode")
        if supported_modes & unsupported_modes:
            raise ValueError("a failure mode cannot be both supported and unavailable")
        expected_labels = [
            MODE_NAMES[mode]
            for mode, output in self.failure_modes.items()
            if output.classification == 1
        ]
        if self.failure_mode_labels != expected_labels:
            raise ValueError("failure_mode_labels contradict the classifier outputs")
        for name, warning in self.out_of_training_range.items():
            if name not in expected - {"Type"}:
                raise ValueError("out_of_training_range contains an unknown feature")
            in_range = warning.training_range[0] <= warning.value <= warning.training_range[1]
            if warning.value != inputs[name] or in_range:
                raise ValueError("out_of_training_range does not match the supplied inputs")
        limitations = self.limitations.lower()
        if "ai4i synthetic-data classification" not in limitations or "not a future failure date" not in limitations:
            raise ValueError("AI4I limitations are missing")
        return self


class ChatRequest(BaseModel):
    question: str
    prediction_context: ProductionPredictionContext | None = None
    model_config = ConfigDict(extra="forbid")


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
    evidence_status: Literal[
        "sources_available", "no_relevant_evidence", "unavailable", "unknown"
    ] = "unknown"


class FeatureVector(BaseModel):
    product_type: str
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float

    class Config:
        extra = "forbid"


class PredictRequest(BaseModel):
    machine_id: str
    observed_at: datetime | None = None
    features: FeatureVector

    class Config:
        extra = "forbid"


class PredictResponse(BaseModel):
    prediction: dict[str, Any]
    evidence: list[dict[str, Any]]
    machine_specific_evidence: list[dict[str, Any]]
    recommendation: str
    evidence_status: str


class PredictiveModelMetadata(BaseModel):
    ready: bool
    model_version: str
    trained_at: str | None = None
    features: list[str]
    feature_ranges: dict[str, list[float]]
    allowed_types: list[str]
    supported_failure_modes: list[str]
    unsupported_failure_modes: dict[str, str]


@app.get("/health")
def health():
    """Report API and corpus readiness instead of only process liveness."""
    global _corpus_status
    try:
        _corpus_status = corpus_stats()
    except Exception:
        pass
    try:
        _predictor.metadata()
        predictor_status = "ready"
    except ModelUnavailable:
        predictor_status = "not_trained"
    except Exception:
        predictor_status = "unavailable"
    return {
        "status": "ok" if _corpus_status.get("documents", 0) else "degraded",
        "corpus": _corpus_status,
        "mysql_credentials_configured": mysql_credentials_available(),
        "ollama_model": OLLAMA_MODEL,
        "ollama_status": _ollama_model_status(),
        "sensor_status": "not_connected",
        "predictive_model": predictor_status,
        "ingestion_error": _ingestion_error,
    }


@app.get("/maintenance/history")
def maintenance_history(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List real maintenance rows already indexed in the local SQLite corpus."""
    try:
        return list_maintenance_records(limit=limit, offset=offset)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Local maintenance history is unavailable.",
        ) from exc


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    started = time.perf_counter()
    timings: dict[str, float | None] = {}
    timing_state = _chat_timing_context.get()
    if timing_state is not None:
        timings["request_validation_parse_ms"] = round(
            (started - timing_state["request_started"]) * 1000, 2
        )
        timing_state["pre_route_validation_parse_ms"] = timings["request_validation_parse_ms"]
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    # Experimental inference is available only behind an explicit command.
    # The adapter itself imports the saved Azure service only after validating
    # that command's JSON request; ordinary chat stays on the existing path.
    from experiments.azure_pdm.chat_adapter import (
        ExperimentalRequestError,
        handle_experimental_chat,
        is_experimental_request,
    )
    if is_experimental_request(question):
        if request.prediction_context is not None:
            raise HTTPException(
                status_code=422,
                detail="Production AI4I context cannot be attached to an experimental prediction request.",
            )
        try:
            result = handle_experimental_chat(
                question,
                retrieve_fn=retrieve,
                generate_fn=generate_answer,
                corpus_ready=bool(_corpus_status.get("documents")),
            )
        except ExperimentalRequestError as exc:
            _log_chat_timing({}, started, len(question), 0, "experimental_request_invalid")
            if timing_state is not None:
                timing_state["route_ms"] = (time.perf_counter() - started) * 1000
            return ChatResponse(answer=f"Experimental prediction request was not accepted: {exc}", sources=[])
        except Exception:
            _log_chat_timing({}, started, len(question), 0, "experimental_evidence_invalid")
            if timing_state is not None:
                timing_state["route_ms"] = (time.perf_counter() - started) * 1000
            return ChatResponse(
                answer="The experimental service returned incomplete or inconsistent evidence. SMARTA did not use it to generate an explanation.",
                sources=[],
            )
        _log_chat_timing({}, started, len(question), result.retrieved_documents, result.outcome)
        if timing_state is not None:
            timing_state["route_ms"] = (time.perf_counter() - started) * 1000
        return ChatResponse(answer=result.answer, sources=list(result.sources))
    prediction_context = request.prediction_context
    if not _corpus_status.get("documents") and prediction_context is None:
        raise HTTPException(status_code=503, detail="SMARTA's document corpus is not ready.")
    results = []
    documentation_status = None
    failed_stage = "retrieval"
    try:
        if prediction_context is not None and not _corpus_status.get("documents"):
            documentation_status = "corpus_unavailable"
        else:
            phase_started = time.perf_counter()
            retrieval_query = question
            if prediction_context is not None:
                labels = [
                    mode for mode, output in prediction_context.failure_modes.items()
                    if output.classification == 1
                ]
                classification = (
                    ", ".join(labels) if labels else "no supported failure mode classified"
                )
                retrieval_query = (
                    f"{question} AI4I machine maintenance records "
                    f"machine {prediction_context.machine_id} {classification}"
                )
            try:
                if (
                    prediction_context is not None
                    and _prediction_question_intent(question) == "troubleshooting"
                ):
                    results = _retrieve_prediction_troubleshooting(question, retrieval_query)
                else:
                    results = retrieve(retrieval_query)
                timings["retrieval_ms"] = round((time.perf_counter() - phase_started) * 1000, 2)
                if not results and prediction_context is not None:
                    documentation_status = "no_matches"
            except Exception:
                if prediction_context is None:
                    raise
                results = []
                documentation_status = "retrieval_failed"
                timings["retrieval_ms"] = round((time.perf_counter() - phase_started) * 1000, 2)
        phase_started = time.perf_counter()
        context_parts = [
            f"[{index}] {item['source']}, {item['locator']}\n{item['text']}"
            for index, item in enumerate(results, start=1)
        ]
        identifiers = re.findall(r"\b[A-Z]{1,5}\d{1,5}[A-Z]?\b", question.upper())
        for identifier in identifiers:
            found_exact = any(
                re.search(rf"(?<![A-Z0-9]){re.escape(identifier)}(?![A-Z0-9])", item["text"].upper())
                for item in results
            )
            if not found_exact:
                context_parts.insert(
                    0,
                    f"SYSTEM NOTE: No retrieved record contains the exact identifier {identifier}.",
                )
        context = "\n\n".join(context_parts)
        timings["context_chars"] = len(context)
        timings["context_assembly_ms"] = round(
            (time.perf_counter() - phase_started) * 1000, 2
        )
        failed_stage = "generation"
        if prediction_context is not None:
            answer = generate_answer(
                context=context,
                question=question,
                timings=timings,
                production_prediction_evidence=prediction_context.model_dump(mode="json"),
                documentation_status=documentation_status or "available",
            )
        else:
            answer = generate_answer(context=context, question=question, timings=timings)
        timings.setdefault(
            "prompt_preparation_ms",
            round((time.perf_counter() - phase_started) * 1000, 2),
        )
    except Exception as exc:
        # Keep provider and retrieval diagnostics in server logs, while exposing
        # only a stable category and safe recovery instruction to the client.
        if isinstance(exc, GenerationFailure):
            code = exc.code
            message = exc.public_message
            status_code = 503 if code == "language_model_unavailable" else (
                504 if code == "generation_timeout" else 502
            )
        elif failed_stage == "retrieval":
            code = "retrieval_failed"
            message = "SMARTA could not search its maintenance records. Please retry."
            status_code = 503
        else:
            code = "generation_failed"
            message = "SMARTA could not generate a response. Please try again."
            status_code = 502
        _logger.exception("chat_request_failed stage=%s code=%s", failed_stage, code)
        _log_chat_timing(
            timings,
            started,
            len(question),
            len(results) if "results" in locals() else 0,
            code,
        )
        if timing_state is not None:
            timing_state["route_ms"] = (time.perf_counter() - started) * 1000
        raise HTTPException(
            status_code=status_code,
            detail={"code": code, "message": message},
        ) from exc

    citations = [
        f"[{index}] {item['source']}, {item['locator']}"
        for index, item in enumerate(results, start=1)
    ]
    _log_chat_timing(timings, started, len(question), len(results), "success")
    if timing_state is not None:
        timing_state["route_ms"] = (time.perf_counter() - started) * 1000
    evidence_status = (
        "unavailable"
        if documentation_status in {"retrieval_failed", "corpus_unavailable"}
        else "sources_available" if results else "no_relevant_evidence"
    )
    return ChatResponse(
        answer=answer, sources=citations, evidence_status=evidence_status
    )


def _log_chat_timing(
    timings: dict[str, float | None],
    started: float,
    question_chars: int,
    retrieved_documents: int,
    outcome: str,
) -> None:
    """Log phase durations without query text, prompt text, or source contents."""
    _logger.info(
        "chat_request_timing %s",
        json.dumps(
            {
                "retrieval_ms": None,
                "query_embedding_ms": None,
                "retrieval_method": "sqlite_fts5_bm25_lexical_rerank",
                "context_assembly_ms": None,
                "prompt_preparation_ms": None,
                "model_generation_ms": None,
                **timings,
                "total_server_ms": round((time.perf_counter() - started) * 1000, 2),
                "question_chars": question_chars,
                "retrieved_documents": retrieved_documents,
                "outcome": outcome,
            },
            sort_keys=True,
        ),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Classify the supplied operating point and return separately sourced RAG evidence."""
    if not request.machine_id.strip():
        raise HTTPException(status_code=422, detail="machine_id cannot be empty.")
    features = {
        "Type": request.features.product_type,
        "Air temperature [K]": request.features.air_temperature_k,
        "Process temperature [K]": request.features.process_temperature_k,
        "Rotational speed [rpm]": request.features.rotational_speed_rpm,
        "Torque [Nm]": request.features.torque_nm,
        "Tool wear [min]": request.features.tool_wear_min,
    }
    try:
        prediction = _predictor.predict(features, machine_id=request.machine_id.strip())
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    positive_modes = [
        MODE_NAMES[mode] for mode, result in prediction["failure_modes"].items()
        if result["classification"] == 1
    ]
    classes = ", ".join(positive_modes) if positive_modes else "no supported failure mode classified"
    query = f"{classes} predictive maintenance " + " ".join(positive_modes)
    try:
        machine_candidates = retrieve(request.machine_id, limit=6)
        general_candidates = retrieve(query, limit=6)
    except Exception:
        machine_candidates, general_candidates = [], []
    machine_pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(request.machine_id)}(?![A-Za-z0-9])", re.IGNORECASE)
    machine_specific = [
        item for item in machine_candidates
        if machine_pattern.search(item["text"] + " " + item["source"] + " " + item["locator"])
    ]
    results = []
    seen_evidence = set()
    for item in machine_specific + general_candidates:
        key = (item["source"], item["locator"])
        if key not in seen_evidence:
            seen_evidence.add(key)
            results.append(item)
        if len(results) == 6:
            break
    evidence_payload = [
        {"source": item["source"], "locator": item["locator"], "text": item["text"]}
        for item in results
    ]
    exact_payload = [
        {"source": item["source"], "locator": item["locator"], "text": item["text"]}
        for item in machine_specific
    ]
    if not results:
        recommendation = "No sufficiently supported corrective action was found in the available maintenance records or documentation. Further inspection is required."
        evidence_status = "insufficient"
    else:
        context_parts = [f"[{i}] {item['source']}, {item['locator']}\n{item['text']}" for i, item in enumerate(results, 1)]
        context = "\n\n".join(context_parts)
        question = (
            f"The ML classifier reports {classes} for machine {request.machine_id}. "
            "Explain this as a model classification, not a confirmed fault or future forecast. "
            "Give only possible checks supported by the excerpts. State explicitly when no corrective action or machine-specific history is supported."
        )
        try:
            recommendation = generate_answer(context, question)
        except Exception:
            recommendation = "Relevant source excerpts were found, but SMARTA could not generate a grounded recommendation. Review the cited evidence and inspect the machine."
        evidence_status = "machine_specific_and_general" if machine_specific else "general_only"
    prediction["observed_at"] = request.observed_at.isoformat() if request.observed_at else None
    prediction["failure_mode_labels"] = positive_modes
    return PredictResponse(
        prediction=prediction,
        evidence=evidence_payload,
        machine_specific_evidence=exact_payload,
        recommendation=recommendation,
        evidence_status=evidence_status,
    )


@app.get("/predict/metadata", response_model=PredictiveModelMetadata)
def predictive_model_metadata():
    """Expose model readiness and the exact supported simulator input ranges."""
    try:
        return _predictor.metadata()
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    # Allows running with `python api.py` from any working directory.
    import uvicorn

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
