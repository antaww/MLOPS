from fastapi import APIRouter

from app.api import health, nlp, stt

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(nlp.router, tags=["nlp"])
api_router.include_router(stt.router, tags=["stt"])
