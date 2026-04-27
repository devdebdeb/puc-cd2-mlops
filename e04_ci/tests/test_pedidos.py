"""
e04 — tests/test_pedidos.py
============================
Exercício 3.5 (e04_p02, Desafio): cobertura de pedidos.

Regras de negócio testadas:
  - Prato inexistente → 404
  - Prato indisponível → 400
  - Quantidade zero → 422 (validação Pydantic)
  - Valor total = preco × quantidade
"""

import pytest


@pytest.mark.smoke
def test_criar_pedido_com_prato_existente(client):
    payload = {
        "prato_id": 1,
        "quantidade": 2,
        "observacao": "sem cebola",
    }
    response = client.post("/pedidos", json=payload)
    assert response.status_code in [200, 201]
    dados = response.json()
    assert "valor_total" in dados
    assert "nome_prato" in dados


@pytest.mark.smoke
def test_valor_total_calculado_corretamente(client):
    """
    Busca o preço unitário real do prato antes de criar o pedido.
    Robusto: não hardcode o preço — usa o contrato da API.
    """
    prato = client.get("/pratos/1").json()
    preco_unitario = prato["preco"]

    payload = {"prato_id": 1, "quantidade": 3}
    response = client.post("/pedidos", json=payload)
    assert response.status_code in [200, 201]
    assert response.json()["valor_total"] == round(preco_unitario * 3, 2)


@pytest.mark.smoke
def test_criar_pedido_com_prato_inexistente_retorna_404(client):
    payload = {"prato_id": 9999, "quantidade": 1}
    response = client.post("/pedidos", json=payload)
    assert response.status_code == 404


@pytest.mark.smoke
def test_criar_pedido_com_prato_indisponivel_retorna_400(client):
    """
    Prato 3 (Lasanha Bolonhesa) está marcado como indisponível nos dados iniciais.
    Alternativamente, usa PUT para garantir estado antes do teste.
    """
    # Garante que o prato 3 está indisponível
    client.put("/pratos/3/disponibilidade", json={"disponivel": False})
    payload = {"prato_id": 3, "quantidade": 1}
    response = client.post("/pedidos", json=payload)
    assert response.status_code == 400


@pytest.mark.smoke
def test_criar_pedido_com_quantidade_zero_retorna_422(client):
    """Validação Pydantic: quantidade mínima é 1 (Field ge=1)."""
    payload = {"prato_id": 1, "quantidade": 0}
    response = client.post("/pedidos", json=payload)
    assert response.status_code == 422
