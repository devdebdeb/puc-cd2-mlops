"""
e04 — tests/test_bebidas.py
============================
Exercício 3.5 (e04_p02, Desafio): cobertura de bebidas.
Segue os mesmos padrões do test_pratos.py.
"""

import pytest


@pytest.mark.smoke
def test_listar_bebidas_retorna_200(client):
    response = client.get("/bebidas")
    assert response.status_code == 200


@pytest.mark.smoke
def test_listar_bebidas_retorna_lista(client):
    response = client.get("/bebidas")
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0


@pytest.mark.smoke
def test_filtro_por_tipo_retorna_apenas_tipo_correto(client):
    response = client.get("/bebidas?tipo=vinho")
    assert response.status_code == 200
    for bebida in response.json():
        assert bebida["tipo"] == "vinho"


@pytest.mark.smoke
def test_filtro_alcoolica_retorna_correto(client):
    response = client.get("/bebidas?alcoolica=false")
    assert response.status_code == 200
    for bebida in response.json():
        assert bebida["alcoolica"] is False


@pytest.mark.smoke
def test_buscar_bebida_existente_retorna_campos_esperados(client):
    response = client.get("/bebidas/1")
    assert response.status_code == 200
    bebida = response.json()
    assert "id" in bebida
    assert "nome" in bebida
    assert "preco" in bebida


@pytest.mark.smoke
def test_buscar_bebida_inexistente_retorna_404(client):
    response = client.get("/bebidas/9999")
    assert response.status_code == 404


@pytest.mark.smoke
def test_criar_bebida_valida(client, bebida_valida):
    response = client.post("/bebidas", json=bebida_valida)
    assert response.status_code in [200, 201]
    dados = response.json()
    assert dados["nome"] == bebida_valida["nome"]
    assert "id" in dados


@pytest.mark.smoke
def test_criar_bebida_com_tipo_invalido_retorna_422(client):
    bebida_invalida = {
        "nome": "Energético XYZ",
        "tipo": "energetico",   # não está no pattern
        "preco": 12.0,
        "alcoolica": False,
        "volume_ml": 250,
    }
    response = client.post("/bebidas", json=bebida_invalida)
    assert response.status_code == 422


@pytest.mark.smoke
def test_criar_bebida_com_volume_invalido_retorna_422(client):
    bebida_invalida = {
        "nome": "Mini Água",
        "tipo": "agua",
        "preco": 5.0,
        "alcoolica": False,
        "volume_ml": 10,  # abaixo de 50 ml
    }
    response = client.post("/bebidas", json=bebida_invalida)
    assert response.status_code == 422


@pytest.mark.smoke
def test_bebida_criada_aparece_na_listagem(client):
    nome_unico = "Prosecco Teste ABC-1234"
    client.post("/bebidas", json={
        "nome": nome_unico,
        "tipo": "vinho",
        "preco": 85.0,
        "alcoolica": True,
        "volume_ml": 750,
    })
    response = client.get("/bebidas")
    nomes = [b["nome"] for b in response.json()]
    assert nome_unico in nomes
