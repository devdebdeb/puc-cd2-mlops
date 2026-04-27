"""
e04 — tests/test_modelo.py
===========================
Exercícios cobertos:
  5.2 (e04_p03) — Baixando o modelo no pipeline (load_model + testes de estrutura)
  5.3 (e04_p03) — Testando o endpoint /ml/predict
  5.5 (e04_p03, Desafio) — Comportamento do modelo (casos extremos + determinismo)

Conceitos aplicados:
  - @pytest.mark.integracao para separar dos smoke tests
  - fixture com scope="module" para evitar download repetido
  - Testes de estrutura (predict retorna 0 ou 1, probability entre 0–1)
  - Testes de comportamento (sanidade: caso suspeito > caso típico)
  - Parametrização para campos inválidos

NOTA: configure a variável de ambiente HF_REPO_ID com o seu repo_id real
      e HF_TOKEN com o seu token antes de rodar estes testes localmente.

      Windows PowerShell:
        $env:HF_REPO_ID = "seu-usuario/mlops-bella-tavola-v1"
        $env:HF_TOKEN   = "hf_..."
        pytest tests/test_modelo.py -v -m integracao
"""

import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app

# ---------------------------------------------------------------------------
# Constantes — ajuste para o seu repositório
# ---------------------------------------------------------------------------
REPO_ID = os.environ.get("HF_REPO_ID")
if not REPO_ID:
    pytest.skip(
        "HF_REPO_ID não configurada. Pulando testes de integração.",
        allow_module_level=True,
    )
N_FEATURES = 5

PAYLOAD_VALIDO = {
    "valor_pedido": 120.0,
    "hora_pedido": 20,
    "num_itens": 3,
    "historico_cancelamentos": 0,
    "distancia_entrega": 2.5,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def modelo():
    """
    Carrega o modelo uma única vez para todos os testes deste módulo.
    scope="module" evita baixar o artefato a cada teste — importante
    para não sobrecarregar o pipeline com downloads repetidos.
    """
    from model_utils import load_model

    return load_model(REPO_ID)


@pytest.fixture(scope="module")
def amostra_valida():
    """
    Amostra com valores plausíveis para o domínio de fraude.
    Valores dentro das faixas do gerar_dataset (seed=42).
    """
    return np.array([[150.0, 14, 500.0, 3, 0]])


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Exercício 5.2 — Testes de estrutura do modelo carregado
# ---------------------------------------------------------------------------


@pytest.mark.integracao
def test_modelo_carregado_nao_e_none(modelo):
    assert modelo is not None


@pytest.mark.integracao
def test_modelo_tem_metodo_predict(modelo):
    assert hasattr(modelo, "predict")
    assert callable(modelo.predict)


@pytest.mark.integracao
def test_modelo_tem_metodo_predict_proba(modelo):
    assert hasattr(modelo, "predict_proba")
    assert callable(modelo.predict_proba)


@pytest.mark.integracao
def test_predict_retorna_array_com_formato_correto(modelo, amostra_valida):
    resultado = modelo.predict(amostra_valida)
    assert resultado.shape == (1,)
    assert resultado[0] in [0, 1]


@pytest.mark.integracao
def test_predict_proba_retorna_probabilidades_validas(modelo, amostra_valida):
    probas = modelo.predict_proba(amostra_valida)
    assert probas.shape == (1, 2)  # duas classes
    assert abs(probas[0].sum() - 1.0) < 1e-6  # soma = 1
    assert all(0 <= p <= 1 for p in probas[0])  # cada valor entre 0 e 1


# ---------------------------------------------------------------------------
# Exercício 5.3 — Testando o endpoint /ml/predict via TestClient
# ---------------------------------------------------------------------------


@pytest.mark.integracao
def test_predict_retorna_200(client):
    response = client.post("/ml/predict", json=PAYLOAD_VALIDO)
    assert response.status_code == 200


@pytest.mark.integracao
def test_predict_retorna_campos_esperados(client):
    response = client.post("/ml/predict", json=PAYLOAD_VALIDO)
    assert response.status_code == 200
    dados = response.json()
    assert "prediction" in dados
    assert "probability" in dados
    assert "label" in dados
    assert "model_version" in dados


@pytest.mark.integracao
def test_predict_prediction_e_binario(client):
    response = client.post("/ml/predict", json=PAYLOAD_VALIDO)
    assert response.json()["prediction"] in [0, 1]


@pytest.mark.integracao
def test_predict_probability_entre_zero_e_um(client):
    response = client.post("/ml/predict", json=PAYLOAD_VALIDO)
    probability = response.json()["probability"]
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0


@pytest.mark.integracao
def test_predict_label_e_string_nao_vazia(client):
    response = client.post("/ml/predict", json=PAYLOAD_VALIDO)
    label = response.json()["label"]
    assert isinstance(label, str)
    assert len(label) > 0


@pytest.mark.integracao
def test_predict_sem_campo_obrigatorio_retorna_422(client):
    payload_incompleto = {"valor_pedido": 120.0}  # faltam outros campos
    response = client.post("/ml/predict", json=payload_incompleto)
    assert response.status_code == 422


@pytest.mark.integracao
@pytest.mark.parametrize(
    "campo,valor_invalido",
    [
        ("hora_pedido", 25),  # hora fora de 0–23
        ("hora_pedido", -1),  # hora negativa
        ("num_itens", 0),  # quantidade inválida (ge=1)
        ("valor_pedido", -50.0),  # valor negativo (gt=0)
    ],
)
def test_predict_campo_invalido_retorna_422(client, campo, valor_invalido):
    payload = {**PAYLOAD_VALIDO, campo: valor_invalido}
    response = client.post("/ml/predict", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Exercício 5.5 (Desafio) — Testes de comportamento do modelo
# ---------------------------------------------------------------------------


@pytest.mark.integracao
def test_modelo_distingue_casos_extremos(client):
    """
    Teste de sanidade: o modelo deve atribuir probabilidade MAIOR
    para um pedido com perfil suspeito do que para um pedido típico.

    Valores construídos a partir da lógica do gerar_dataset (e03):
    - caso_suspeito: valor alto, madrugada, distância grande → alta chance de fraude
    - caso_tipico:   valor baixo, horário comercial, perto    → baixa chance

    AVISO: se este teste falhar após retreinamento, investigue antes de
    ajustar os valores — pode indicar que o modelo mudou de comportamento.
    """
    caso_tipico = {
        "valor_pedido": 55.0,
        "hora_pedido": 13,
        "num_itens": 2,
        "historico_cancelamentos": 0,
        "distancia_entrega": 1.5,
    }
    caso_suspeito = {
        "valor_pedido": 8_900.0,
        "hora_pedido": 2,
        "num_itens": 9,
        "historico_cancelamentos": 4,
        "distancia_entrega": 4_500.0,
    }

    resp_tipico = client.post("/ml/predict", json=caso_tipico)
    resp_suspeito = client.post("/ml/predict", json=caso_suspeito)

    assert resp_tipico.status_code == 200
    assert resp_suspeito.status_code == 200

    prob_tipico = resp_tipico.json()["probability"]
    prob_suspeito = resp_suspeito.json()["probability"]

    assert prob_suspeito > prob_tipico, (
        f"Esperado: prob_suspeito ({prob_suspeito:.3f}) > "
        f"prob_tipico ({prob_tipico:.3f})\n"
        f"Se falhar após retreinamento, investigue antes de ajustar o teste."
    )


@pytest.mark.integracao
def test_modelo_e_deterministico(client):
    """O mesmo input deve sempre gerar o mesmo resultado."""
    resp_1 = client.post("/ml/predict", json=PAYLOAD_VALIDO)
    resp_2 = client.post("/ml/predict", json=PAYLOAD_VALIDO)
    assert resp_1.json()["prediction"] == resp_2.json()["prediction"]
    assert resp_1.json()["probability"] == resp_2.json()["probability"]
