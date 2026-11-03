"""Ponto de entrada da API.

Execução (na raiz do projeto):
    uvicorn app.main:app --reload
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core import config
from app.services import model_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo na RAM uma única vez, na inicialização.

    As rotas apenas leem app.state.model_bundle: nenhuma requisição toca o disco.
    Se o carregamento falhar, a API sobe mesmo assim e o /health informa o problema.
    """
    try:
        app.state.model_bundle = model_service.load_model(config.MODEL_PATH)
    except Exception:
        logger.exception("Falha ao carregar o modelo; a API iniciará sem modelo.")
        app.state.model_bundle = None
    yield
    # Shutdown: libera a referência ao modelo
    app.state.model_bundle = None
    logger.info("Aplicação finalizada.")


app = FastAPI(lifespan=lifespan)