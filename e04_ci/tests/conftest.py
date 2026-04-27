"""
e04 — tests/conftest.py
========================
Fixtures compartilhadas para toda a suíte de testes.

Conceito (e04_p02, Bloco 4, ex 4.1):
  Centralizar fixtures no conftest.py elimina a variável global 'client'
  dos arquivos de teste e permite injeção automática pelo pytest.

  IMPORTANTE: a fixture 'client' recria o TestClient a cada teste (scope padrão
  = function), mas NÃO reinicializa o estado global da API (listas de pratos,
  bebidas, pedidos). Para testes robustos, use comportamentos relativos —
  não contagens absolutas (ver ex 4.2).
"""

import pytest
from fastapi.testclient import TestClient

from main import app  # importa a instância FastAPI do main.py


@pytest.fixture
def client():
    """
    Cria um TestClient para cada função de teste.
    O scope 'function' (padrão) garante um client novo por teste,
    mas o estado interno da aplicação persiste entre eles.
    """
    return TestClient(app)


@pytest.fixture
def prato_valido():
    """Payload de prato válido reutilizável nos testes de POST."""
    return {
        "nome": "Prato de Fixture",
        "categoria": "massa",
        "preco": 45.0,
        "disponivel": True,
    }


@pytest.fixture
def bebida_valida():
    """Payload de bebida válida reutilizável nos testes de POST."""
    return {
        "nome": "Água de Fixture",
        "tipo": "agua",
        "preco": 8.0,
        "alcoolica": False,
        "volume_ml": 500,
    }
