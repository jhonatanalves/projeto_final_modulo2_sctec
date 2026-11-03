"""Rotas de inferência do modelo."""
from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.prediction import PredictRequest, PredictResponse
from app.services import model_service
from app.services.model_service import BusinessRuleError

router = APIRouter(prefix="/v1", tags=["Predição"])


@router.post("/predict", response_model=PredictResponse, summary="Prevê o churn de um cliente")
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Recebe os dados já validados pelo Pydantic e delega a inferência ao service.

    A rota não contém regra de modelo: ela só traduz HTTP <-> camada de serviço.
    """
    bundle = request.app.state.model_bundle  # modelo já em memória (carregado no lifespan)
    try:
        return model_service.predict(bundle, payload)
    except BusinessRuleError as exc:
        # Tipos e faixas corretos (passou no 422), mas combinação incoerente -> 400
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc