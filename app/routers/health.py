"""Rota de monitoramento do serviço."""
from fastapi import APIRouter, Request

from app.core import config
from app.schemas.prediction import HealthResponse

router = APIRouter(tags=["Monitoramento"])


@router.get("/health", response_model=HealthResponse, summary="Verifica a saúde do serviço")
def health(request: Request) -> HealthResponse:
    """Sempre responde 200 enquanto a API estiver no ar.

    O campo `model_loaded` indica se o modelo está na memória RAM (lido de
    app.state, sem acessar o disco).
    """
    bundle = getattr(request.app.state, "model_bundle", None)
    return HealthResponse(
        status="ok" if bundle is not None else "degraded",
        model_loaded=bundle is not None,
        model_version=bundle.get("model_version") if bundle else None,
        api_version=config.API_VERSION,
    )