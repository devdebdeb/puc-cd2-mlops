"""
e04 — tests/test_pratos.py
===========================
Exercícios cobertos:
  3.3 (e04_p02) — Primeiro teste real da API (GET /pratos, GET /pratos/{id})
  3.4 (e04_p02) — Testando criação e validação (POST /pratos)
  4.2 (e04_p02) — Testes robustos vs frágeis (comportamentos relativos)
  4.3 (e04_p02) — Parametrização de casos de erro

Conceitos aplicados:
  - TestClient via fixture do conftest.py (ex 4.1)
  - Testes que verificam comportamento relativo, não estado absoluto (ex 4.2)
  - @pytest.mark.parametrize para múltiplos inputs (ex 4.3)
  - @pytest.mark.smoke para filtrar no pipeline
"""

import pytest

# ===========================================================================
# Bloco 3, Exercício 3.3 — Primeiro teste real da API
# ===========================================================================


@pytest.mark.smoke
def test_raiz_retorna_nome_restaurante(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Bella Tavola" in response.json()["restaurante"]


@pytest.mark.smoke
def test_listar_pratos_retorna_200(client):
    response = client.get("/pratos")
    assert response.status_code == 200


@pytest.mark.smoke
def test_listar_pratos_retorna_lista(client):
    response = client.get("/pratos")
    assert isinstance(response.json(), list)


@pytest.mark.smoke
def test_listar_pratos_retorna_pelo_menos_um_prato(client):
    response = client.get("/pratos")
    assert len(response.json()) > 0


@pytest.mark.smoke
def test_filtro_por_categoria_retorna_apenas_categoria_correta(client):
    response = client.get("/pratos?categoria=pizza")
    assert response.status_code == 200
    for prato in response.json():
        assert prato["categoria"] == "pizza"


@pytest.mark.smoke
def test_buscar_prato_existente_retorna_campos_esperados(client):
    response = client.get("/pratos/1")
    assert response.status_code == 200
    prato = response.json()
    assert "id" in prato
    assert "nome" in prato
    assert "preco" in prato


@pytest.mark.smoke
def test_buscar_prato_inexistente_retorna_404(client):
    """
    Exercício 3.3 + 2.2 (e02):
    HTTPException deve retornar 404 (não mais 200 com mensagem).
    """
    response = client.get("/pratos/9999")
    assert response.status_code == 404


# ===========================================================================
# Bloco 3, Exercício 3.4 — Testando criação e validação
# ===========================================================================


@pytest.mark.smoke
def test_criar_prato_valido(client, prato_valido):
    response = client.post("/pratos", json=prato_valido)
    assert response.status_code in [200, 201]
    dados = response.json()
    assert dados["nome"] == prato_valido["nome"]
    assert dados["preco"] == prato_valido["preco"]
    assert "id" in dados


@pytest.mark.smoke
def test_criar_prato_com_preco_negativo_retorna_422(client):
    prato_invalido = {
        "nome": "Prato Inválido",
        "categoria": "pizza",
        "preco": -10.0,
    }
    response = client.post("/pratos", json=prato_invalido)
    assert response.status_code == 422


@pytest.mark.smoke
def test_criar_prato_com_nome_curto_retorna_422(client):
    prato_invalido = {
        "nome": "AB",  # menos de 3 caracteres
        "categoria": "pizza",
        "preco": 40.0,
    }
    response = client.post("/pratos", json=prato_invalido)
    assert response.status_code == 422


@pytest.mark.smoke
def test_criar_prato_com_categoria_invalida_retorna_422(client):
    prato_invalido = {
        "nome": "Prato Exótico",
        "categoria": "esoterico",
        "preco": 40.0,
    }
    response = client.post("/pratos", json=prato_invalido)
    assert response.status_code == 422


@pytest.mark.smoke
def test_prato_criado_aparece_na_listagem(client):
    """
    Exercício 3.4 + 4.2 (robusto):
    Usa nome único para não depender de contagem absoluta.
    """
    nome_unico = "Tagliatelle Teste XYZ-9871"
    client.post(
        "/pratos",
        json={
            "nome": nome_unico,
            "categoria": "massa",
            "preco": 68.0,
        },
    )
    response = client.get("/pratos")
    nomes = [p["nome"] for p in response.json()]
    assert nome_unico in nomes


# ===========================================================================
# Bloco 4, Exercício 4.2 — Testes robustos vs frágeis (versão robusta)
# ===========================================================================


@pytest.mark.smoke
def test_lista_retorna_pratos_com_estrutura_correta(client):
    """
    Robusto: verifica estrutura de cada item, não contagem absoluta.
    (versão corrigida do teste frágil 'assert len == 6')
    """
    response = client.get("/pratos")
    assert response.status_code == 200
    pratos = response.json()
    assert len(pratos) > 0
    assert "id" in pratos[0]
    assert "nome" in pratos[0]
    assert "preco" in pratos[0]


@pytest.mark.smoke
def test_margherita_esta_no_cardapio(client):
    """Robusto: verifica presença por nome, não por posição."""
    response = client.get("/pratos")
    nomes = [p["nome"] for p in response.json()]
    assert "Margherita" in nomes


@pytest.mark.smoke
def test_novo_prato_recebe_id_valido(client, prato_valido):
    """Robusto: verifica que o ID é int positivo, não um valor absoluto."""
    response = client.post("/pratos", json=prato_valido)
    assert response.status_code in [200, 201]
    dados = response.json()
    assert "id" in dados
    assert isinstance(dados["id"], int)
    assert dados["id"] > 0


@pytest.mark.smoke
def test_filtro_categoria_retorna_apenas_categoria_correta(client):
    """Robusto: verifica que TODOS os retornados são pizza."""
    response = client.get("/pratos?categoria=pizza")
    assert response.status_code == 200
    for prato in response.json():
        assert prato["categoria"] == "pizza"


# ===========================================================================
# Bloco 4, Exercício 4.3 — Parametrização
# ===========================================================================


@pytest.mark.parametrize(
    "categoria_invalida",
    [
        "esoterico",
        "fastfood",
        "japonesa",
        "PIZZA",  # case-sensitive — não é igual a 'pizza'
        "massa extra",  # espaço não permitido pelo pattern regex
    ],
)
def test_categoria_invalida_retorna_422(client, categoria_invalida):
    prato = {
        "nome": "Prato Teste",
        "categoria": categoria_invalida,
        "preco": 40.0,
    }
    response = client.post("/pratos", json=prato)
    assert response.status_code == 422


@pytest.mark.parametrize("id_inexistente", [9999, 123456, 99999])
def test_prato_inexistente_retorna_404(client, id_inexistente):
    """IDs sintaticamente válidos mas inexistentes → 404."""
    response = client.get(f"/pratos/{id_inexistente}")
    assert response.status_code == 404


@pytest.mark.parametrize(
    "categoria_valida",
    [
        "pizza",
        "massa",
        "sobremesa",
    ],
)
def test_filtro_categoria_valida(client, categoria_valida):
    """Verifica que o filtro funciona corretamente para cada categoria válida."""
    response = client.get(f"/pratos?categoria={categoria_valida}")
    assert response.status_code == 200
    for prato in response.json():
        assert prato["categoria"] == categoria_valida
