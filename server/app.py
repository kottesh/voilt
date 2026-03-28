"""FastAPI ingest application."""

from __future__ import annotations

import json

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

from server.services.storage import InMemoryStore
from server.services.verification import verify_event
from shared.schemas import EvidenceRef, ViolationEvent

app = FastAPI(title="voilt-ingest")
store = InMemoryStore()


@app.get("/health")
def health() -> dict[str, str]:
    """Health endpoint for probes."""

    return {"status": "ok"}


@app.post("/ingest")
async def ingest_event(
    event_json: str = Form(default=""),
    evidence_0: UploadFile | None = File(default=None),
    evidence_1: UploadFile | None = File(default=None),
    evidence_2: UploadFile | None = File(default=None),
    evidence_3: UploadFile | None = File(default=None),
    evidence_4: UploadFile | None = File(default=None),
    x_idempotency_key: str | None = Header(default=None),
) -> dict[str, str]:
    """Accept edge events, validate idempotency key, and run verification."""

    if not event_json:
        raise HTTPException(status_code=400, detail="missing event_json")
    try:
        payload = ViolationEvent.model_validate(json.loads(event_json))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid event_json") from exc

    if x_idempotency_key is None or x_idempotency_key != payload.idempotency_key:
        raise HTTPException(status_code=400, detail="invalid idempotency key")

    uploaded = [
        item for item in [evidence_0, evidence_1, evidence_2, evidence_3, evidence_4] if item
    ]
    if uploaded:
        refs: list[EvidenceRef] = []
        for upload in uploaded:
            assert upload is not None
            content = await upload.read()
            path = store.save_evidence(
                str(payload.event_id), upload.filename or "evidence.jpg", content
            )
            refs.append(EvidenceRef(kind="uploaded_image", uri=path, score=payload.max_confidence))
        payload.evidence.extend(refs)

    inserted = store.store_raw(payload)
    if not inserted:
        return {"event_id": str(payload.event_id), "status": "duplicate"}
    verified = verify_event(payload)
    store.store_verified(verified)
    return {"event_id": str(payload.event_id), "status": verified.status.value}
