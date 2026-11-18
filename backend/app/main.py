import json
import ipaddress
import logging
import math
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from .config import FRONTEND_ORIGINS
from .database import initialize, log_event, fetch_events, connect
from .ml import registry, FEATURES

logger = logging.getLogger("neurolink.vision")
@asynccontextmanager
async def lifespan(_app):
    initialize()
    registry.load()
    yield

app = FastAPI(title="NEUROLINK 2.0 API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=FRONTEND_ORIGINS, allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type"], allow_credentials=False)

class EventInput(BaseModel):
    kind: str = Field(min_length=1, max_length=32)
    label: str = Field(min_length=1, max_length=120)
    confidence: float | None = Field(default=None, ge=0, le=1)
    latency_ms: float | None = Field(default=None, ge=0)
    session_id: str | None = Field(default=None, max_length=100)
    metadata: dict = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def bounded_metadata(cls, value):
        encoded = json.dumps(value, ensure_ascii=False)
        if len(encoded.encode("utf-8")) > 4096:
            raise ValueError("metadata must be 4 KB or smaller")
        if any(len(str(key)) > 80 for key in value):
            raise ValueError("metadata keys must be 80 characters or shorter")
        return value

class PredictInput(BaseModel):
    features: list[float]
    @field_validator("features")
    @classmethod
    def finite_features(cls, values):
        if any(not math.isfinite(value) for value in values): raise ValueError("Features must be finite numbers")
        return values

class VisionInput(BaseModel):
    kind: Literal["gesture", "eye"]
    features: list[float]
    session_id: str | None = Field(default=None, max_length=100)
    frame_id: int | None = Field(default=None, ge=0)
    @model_validator(mode="after")
    def validate_feature_size(self):
        if len(self.features) != FEATURES[self.kind]: raise ValueError(f"Expected {FEATURES[self.kind]} features for {self.kind}")
        if any(not math.isfinite(value) for value in self.features): raise ValueError("Features must be finite numbers")
        return self

@app.get("/api/health")
def health(): return {"status":"online", "time":datetime.now(timezone.utc).isoformat()}

@app.get("/api/system/status")
def system_status(): return {"api":"online", "database":"online", "models":registry.status()}

@app.get("/api/models/status")
def model_status(): return registry.status()

@app.post("/api/gesture/predict")
def gesture_predict(body: PredictInput):
    if len(body.features) != FEATURES["gesture"]: raise HTTPException(422, "Expected 63 gesture features")
    return registry.predict("gesture", body.features)

@app.post("/api/eye/predict")
def eye_predict(body: PredictInput):
    if len(body.features) != FEATURES["eye"]: raise HTTPException(422, "Expected 24 eye features")
    return registry.predict("eye", body.features)

@app.post("/api/events", status_code=201)
def create_event(body: EventInput):
    values=body.model_dump(); values["metadata"]=json.dumps(values["metadata"])
    event_id = log_event(**values)
    return {"id":event_id, **body.model_dump()}

@app.delete("/api/events")
def clear_local_events(request: Request):
    """Clear analytics only when called from a loopback client."""
    client_host = request.client.host if request.client else ""
    try:
        is_loopback = ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise HTTPException(403, "Event deletion is restricted to loopback clients")
    with connect() as db:
        deleted = db.execute("DELETE FROM events").rowcount
    return {"deleted": deleted}

@app.get("/api/events")
def events(limit: int = 100, kind: str | None = None):
    if not 1 <= limit <= 500: raise HTTPException(422, "limit must be between 1 and 500")
    return fetch_events(limit, kind)

@app.get("/api/analytics/summary")
def analytics_summary():
    with connect() as db:
        rows = db.execute("SELECT kind, COUNT(*) count, AVG(latency_ms) avg_latency, AVG(confidence) avg_confidence FROM events GROUP BY kind").fetchall()
        total = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    grouped = {r["kind"]: {"count":r["count"], "avg_latency_ms":r["avg_latency"], "avg_confidence":r["avg_confidence"]} for r in rows}
    return {"total_interactions":total, "by_kind":grouped, "models":registry.status()}

@app.get("/api/analytics")
def analytics(): return {"summary":analytics_summary(), "events":fetch_events(200)}

@app.websocket("/ws/vision")
async def vision_socket(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            try:
                message=await ws.receive_json()
                body=VisionInput.model_validate(message)
            except (ValidationError, ValueError, TypeError) as exc:
                session_id=message.get("session_id") if isinstance(message,dict) else None
                log_event("inference", "failed", session_id=session_id, metadata=json.dumps({"message":str(exc)}))
                await ws.send_json({"status":"error", "message":str(exc)}); continue
            try:
                if body.kind == "gesture":
                    logger.info("[NEUROLINK BACKEND] Received gesture frame=%s feature_count=%s finite=%s session=%s",
                                body.frame_id, len(body.features), all(math.isfinite(value) for value in body.features), body.session_id)
                result = registry.predict(body.kind, body.features)
                if body.frame_id is not None: result["frame_id"] = body.frame_id
                if result["available"]:
                    logger.info("[NEUROLINK INFERENCE] Model completed kind=%s frame=%s prediction=%s confidence=%.4f latency_ms=%.3f",
                                body.kind, body.frame_id, result["prediction"], result["confidence"], result["latency_ms"])
                    log_event(body.kind, result["prediction"], result["confidence"], result["latency_ms"], body.session_id)
                else:
                    logger.warning("[NEUROLINK INFERENCE] Model unavailable kind=%s status=%s", body.kind, result.get("status"))
                await ws.send_json(result)
            except (ValueError, TypeError, RuntimeError) as exc:
                log_event("inference", "failed", session_id=body.session_id, metadata=json.dumps({"message":str(exc)}))
                await ws.send_json({"status":"error", "message":str(exc)})
    except WebSocketDisconnect: return
