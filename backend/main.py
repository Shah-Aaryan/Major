"""FastAPI application entry point — all routes registered here."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on sys.path so all local modules resolve correctly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Routes
from backend.routes.health import router as health_router
from backend.routes.research import router as research_router
from backend.routes.trading import router as trading_router
from backend.routes.analysis import router as analysis_router
from backend.routes.backtest import router as backtest_router
from backend.routes.optimization import router as optimization_router
from backend.routes.features import router as features_router
from backend.routes.data import router as data_router
from backend.routes.audit import router as audit_router
from backend.routes.parser import router as parser_router
from backend.utils.logging import configure_logging


configure_logging()

app = FastAPI(
    title="ML Trading Research Backend",
    version="2.0.0",
    description=(
        "FastAPI backend for the ML & Quantitative Trading Research project.\n\n"
        "## Modules\n"
        "- **Research** – run ML optimisation pipelines in the background\n"
        "- **Trading** – paper trading session management\n"
        "- **Analysis** – condition analysis, failure detection, comparison reports, "
        "explainability, and leakage checking\n"
        "- **Backtesting** – standalone backtests and walk-forward validation\n"
        "- **Optimization** – 15 optimisers via the optimizer registry\n"
        "- **Features** – 52-indicator registry and feature summary generation\n"
        "- **Data** – list CSV files, data quality reports, session listing\n"
        "- **Audit** – audit trail reading and integrity verification\n"
    ),
)

# CORS – allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health (no prefix)
app.include_router(health_router)

# Domain routers
app.include_router(research_router)
app.include_router(trading_router)
app.include_router(analysis_router)
app.include_router(backtest_router)
app.include_router(optimization_router)
app.include_router(features_router)
app.include_router(data_router)
app.include_router(audit_router)
app.include_router(parser_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


