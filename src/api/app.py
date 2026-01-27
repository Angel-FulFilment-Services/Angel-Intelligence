"""
Angel Intelligence - FastAPI Application

Main API application with authentication and all routes.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.api.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    settings = get_settings()
    logger.info(f"Starting Angel Intelligence API in {settings.angel_env} mode")
    
    # Preload chat model if enabled (only for local models, not external APIs)
    if settings.preload_chat_model and settings.worker_mode in ["interactive", "both"]:
        if settings.llm_api_url:
            logger.info(f"Using external LLM API at {settings.llm_api_url} - no local model to preload")
        else:
            logger.info("Preloading chat model to eliminate first-request delay...")
            try:
                from src.services.interactive import get_interactive_service
                service = get_interactive_service()
                service._ensure_model_loaded()
                logger.info("Chat model preloaded successfully")
            except Exception as e:
                logger.warning(f"Failed to preload chat model: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Angel Intelligence API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Angel Intelligence",
        description="AI-powered call transcription and analysis service for Angel Fulfilment Services",
        version="1.0.0",
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else ["https://pulse.angelfs.co.uk"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes at /api/v1
    app.include_router(router, prefix="/api/v1")
    
    # Also include at root for backwards compatibility
    # Frontend may use either /api/v1/... or /... paths
    app.include_router(router)
    
    return app


# Create default app instance
app = create_app()
