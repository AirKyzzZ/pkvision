"""PkVision FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import analysis, runs, tricks, submissions

app = FastAPI(
    title="PkVision",
    description=(
        "See Every Move. Score Every Trick. "
        "AI-powered parkour trick detection and scoring system. "
        "Analyzes video to identify tricks, score top 3 by difficulty, "
        "and generate explainable audit trails."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])
app.include_router(runs.router, prefix="/api/v1", tags=["Runs"])
app.include_router(tricks.router, prefix="/api/v1", tags=["Tricks"])
app.include_router(submissions.router, prefix="/api/v1", tags=["Submissions"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "pkvision"}
