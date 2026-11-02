"""Camada de serviço: carregamento do modelo e inferência.

Não conhece nada de HTTP (nenhum import do FastAPI). Assim, pode ser testada
e reutilizada sem a API, e as rotas ficam responsáveis só pelo protocolo.
"""
import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


def load_model(path: Path) -> dict:
    """Lê o artefato serializado do disco.

    Deve ser chamada UMA única vez, no lifespan da aplicação.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Artefato não encontrado em '{path}'. Rode 'python scripts/train.py' para gerá-lo."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("Artefato inválido: esperado um dicionário com a chave 'pipeline'.")
    logger.info("Modelo '%s' carregado de %s", bundle.get("model_version"), path.name)
    return bundle

"""Camada de serviço: carregamento do modelo e inferência.

Não conhece nada de HTTP (nenhum import do FastAPI). Assim, pode ser testada
e reutilizada sem a API, e as rotas ficam responsáveis só pelo protocolo.
"""
import logging
from pathlib import Path

import joblib
import pandas as pd

from app.schemas.prediction import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

# Campo da API (snake_case) -> coluna usada no treino (nome original da base)
FEATURE_MAP = {
    "tenure": "tenure",
    "monthly_charges": "MonthlyCharges",
    "total_charges": "TotalCharges",
    "senior_citizen": "SeniorCitizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "internet_service": "InternetService",
    "online_security": "OnlineSecurity",
    "tech_support": "TechSupport",
    "contract": "Contract",
    "paperless_billing": "PaperlessBilling",
    "payment_method": "PaymentMethod",
}


def load_model(path: Path) -> dict:
    """Lê o artefato serializado do disco.

    Deve ser chamada UMA única vez, no lifespan da aplicação.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Artefato não encontrado em '{path}'. Rode 'python scripts/train.py' para gerá-lo."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("Artefato inválido: esperado um dicionário com a chave 'pipeline'.")
    logger.info("Modelo '%s' carregado de %s", bundle.get("model_version"), path.name)
    return bundle


def _to_dataframe(data: PredictRequest, feature_columns: list[str]) -> pd.DataFrame:
    """Converte o schema validado em um DataFrame de uma linha, na ordem do treino."""
    row = {FEATURE_MAP[field]: value for field, value in data.model_dump(exclude={"customer_id"}).items()}
    row["SeniorCitizen"] = int(row["SeniorCitizen"])  # no treino a coluna é 0/1
    return pd.DataFrame([row], columns=feature_columns)


def _risk_level(probability: float) -> str:
    if probability < 0.30:
        return "baixo"
    if probability < 0.60:
        return "médio"
    return "alto"


def predict(bundle: dict, data: PredictRequest) -> PredictResponse:
    """Executa a inferência usando o pipeline já carregado em memória."""
    features = _to_dataframe(data, bundle["feature_columns"])
    probability = float(bundle["pipeline"].predict_proba(features)[0, 1])
    return PredictResponse(
        customer_id=data.customer_id,
        churn=probability >= bundle.get("threshold", 0.5),
        churn_probability=round(probability, 4),
        risk_level=_risk_level(probability),
        model_version=bundle["model_version"],
    )