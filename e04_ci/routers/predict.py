"""
e04 — routers/predict.py
========================
Endpoint /ml/predict — integra o modelo treinado no e03 à API Bella Tavola.

Conceitos (e04_p03):
- Separação entre routers e main.py (modularidade)
- PredictInput define o contrato das features (ordem = contrato do modelo!)
- predict_proba retorna probabilidade da classe positiva (fraude)
- Carregamento lazy do modelo (apenas quando a rota é chamada)

ATENÇÃO: ajuste REPO_ID para o seu repositório real no Hugging Face Hub.
"""

import os

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from model_utils import load_model

router = APIRouter()


# ---------------------------------------------------------------------------
# Constantes — ajuste REPO_ID após publicar o modelo
# ---------------------------------------------------------------------------
def get_repo_id():
    repo_id = os.environ.get("HF_REPO_ID")
    if not repo_id:
        raise RuntimeError("Variável de ambiente HF_REPO_ID não configurada.")
    return repo_id


MODEL_VERSION = "1.0.0"

# Cache em memória para evitar download a cada predição
_model_cache = None


def get_model():
    """Carrega o modelo uma vez e cacheia em memória."""
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model(get_repo_id())
    return _model_cache


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------
class PredictInput(BaseModel):
    """
    Features do pedido na MESMA ORDEM em que são montadas para o modelo.
    Essa ordem deve corresponder exatamente à função gerar_dataset do e03.

    Features (ordem):
    1. valor_pedido              → equivalente a valor_transacao
    2. hora_pedido               → hora do pedido (0–23)
    3. num_itens                 → equivalente a tentativas_senha (proxy de volume)
    4. historico_cancelamentos   → equivalente a tentativas_senha (proxy de risco)
    5. distancia_entrega         → equivalente a distancia_ultima_compra
    """

    valor_pedido: float = Field(gt=0, description="Valor total do pedido em reais")
    hora_pedido: int = Field(ge=0, le=23, description="Hora do pedido (0–23)")
    num_itens: int = Field(ge=1, description="Quantidade de pratos no pedido")
    historico_cancelamentos: int = Field(
        ge=0, description="Cancelamentos anteriores do cliente"
    )
    distancia_entrega: float = Field(ge=0.0, description="Distância de entrega em km")


class PredictOutput(BaseModel):
    prediction: int
    probability: float
    label: str
    model_version: str


# ---------------------------------------------------------------------------
# Rota
# ---------------------------------------------------------------------------
@router.post("/predict", response_model=PredictOutput)
async def predict(data: PredictInput):
    """
    Prediz se um pedido tem risco de ser problemático.

    - prediction: 0 (sem risco) ou 1 (risco alto)
    - probability: probabilidade de risco alto
    - label: descrição textual
    - model_version: versão do modelo em uso
    """
    try:
        model = get_model()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo indisponível: {exc}",
        )

    # Monta o array NA MESMA ORDEM que as features foram geradas no treino
    features = np.array(
        [
            [
                data.valor_pedido,
                data.hora_pedido,
                data.distancia_entrega,  # distancia_ultima_compra no treino
                data.num_itens,  # tentativas_senha no treino (proxy)
                data.historico_cancelamentos,  # pais_diferente no treino (proxy binário)
            ]
        ]
    )

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    label = "Risco alto" if prediction == 1 else "Sem risco"

    return PredictOutput(
        prediction=prediction,
        probability=round(probability, 4),
        label=label,
        model_version=MODEL_VERSION,
    )
