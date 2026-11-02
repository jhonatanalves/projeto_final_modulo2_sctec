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