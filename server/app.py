"""FastAPI ingest application."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from server.api.ingest import router as ingest_router
from server.api.process import router as process_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)

app = FastAPI(
    title="Traffic Violation Server",
    description="Ingest edge-camera frames, run vision analysis, persist confirmed violations.",
    version="0.1.0",
)

app.include_router(ingest_router)
app.include_router(process_router)


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}
