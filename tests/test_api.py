"""Testes da API com o TestClient do FastAPI.

Rodar na raiz do projeto:  pytest -v
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.prediction import EXEMPLO_ENTRADA


@pytest.fixture(scope="module")
def client():
    # O "with" dispara o lifespan, então o modelo é carregado como em produção.
    # raise_server_exceptions=False deixa o handler de erro 500 responder normalmente.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def payload(**alteracoes) -> dict:
    """Cópia do exemplo válido com alguns campos alterados."""
    return {**EXEMPLO_ENTRADA, **alteracoes}


def test_health_retorna_200_com_modelo_carregado(client):
    resposta = client.get("/health")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "ok"
    assert corpo["model_loaded"] is True


def test_predict_valido_retorna_200(client):
    resposta = client.post("/v1/predict", json=payload())
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"customer_id", "churn", "churn_probability", "risk_level", "model_version"}
    assert 0 <= corpo["churn_probability"] <= 1
    assert corpo["risk_level"] in {"baixo", "médio", "alto"}
    assert corpo["customer_id"] == EXEMPLO_ENTRADA["customer_id"]