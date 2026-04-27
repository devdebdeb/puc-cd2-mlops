"""
e04 — tests/test_saude.py
==========================
Exercício 3.1 (e04_p02): estrutura mínima para o pytest funcionar no pipeline.
Exercício 3.2: teste que falha propositalmente (removido após observação).
"""

import pytest


@pytest.mark.smoke
def test_pytest_funcionando():
    """Confirma que o pytest encontrou e executou este arquivo."""
    assert 1 + 1 == 2
