"""FastAPI ingest application."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.api.ingest import router as ingest_router
from server.api.process import router as process_router
from server.routers.violations import router as violations_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)

app = FastAPI(
    title="Traffic Violation Server",
    description="Ingest edge-camera frames, run vision analysis, persist confirmed violations.",
    version="0.1.0",
)

# Configure CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(process_router)
app.include_router(violations_router)

# Create storage directory if it doesn't exist
os.makedirs("storage", exist_ok=True)

# Mount static files directory at /images
app.mount("/images", StaticFiles(directory="storage"), name="images")


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}
