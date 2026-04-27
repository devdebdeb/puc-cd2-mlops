"""
e03 — Bloco 1: Geração de dados sintéticos com sabor de domínio
================================================================
Matéria  : CDIA CD2 2026
Conceitos: make_classification, reprodutibilidade (random_state/seed),
           features nomeadas, distribuições intencionais com NumPy,
           interface limpa com docstring e validação de parâmetros

Domínio escolhido: DETECÇÃO DE FRAUDE
Features escolhidas:
  - valor_transacao    : transações fraudulentas tendem a ter valor mais alto
  - hora_transacao     : fraudes costumam ocorrer de madrugada (0–5h)
  - distancia_ultima_compra: grande distância geográfica é sinal suspeito
  - tentativas_senha   : muitas tentativas indicam ataque de força bruta
  - pais_diferente     : compra em país diferente do habitual

Por que essas features fazem sentido no domínio:
  1. Transações fraudulentas têm perfil diferente das legítimas em valor e horário —
     padrões muito bem documentados na literatura de fraud detection.
  2. A combinação de distância geográfica, tentativas de senha e país diferente
     captura comportamentos de comprometimento de conta (account takeover).
"""

from typing import Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Exercício 1.1 — Referência com make_classification
# ---------------------------------------------------------------------------
from sklearn.datasets import make_classification  # noqa: F401 (mantido para referência)

# ---------------------------------------------------------------------------
# Exercício 1.2 — Função geradora com sabor de domínio (fraude)
# Exercício 1.3 — Versão robusta: proporcao_positivos, ValueError, retorna (df, X, y)
# ---------------------------------------------------------------------------


def gerar_dataset(
    n_samples: int = 1000,
    seed: int = 42,
    proporcao_positivos: float = 0.3,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Gera dataset sintético de detecção de fraude.

    Parâmetros
    ----------
    n_samples : int
        Número de amostras a gerar.
    seed : int
        Seed para reprodutibilidade.
        Em MLOps, fixar o seed é essencial para comparar experimentos
        de forma justa — sem ele, não sabemos se um modelo melhorou de
        verdade ou apenas recebeu dados mais fáceis numa execução.
    proporcao_positivos : float
        Proporção da classe positiva (fraude). Deve estar entre 0.05 e 0.95.

    Retorna
    -------
    df : pd.DataFrame
        Dataset completo com features e target.
    X : np.ndarray
        Matriz de features (shape: n_samples × 5).
    y : np.ndarray
        Vetor de targets binários (0 = legítimo, 1 = fraude).

    Levanta
    -------
    ValueError
        Se proporcao_positivos não estiver no intervalo [0.05, 0.95].

    Exemplo
    -------
    >>> df, X, y = gerar_dataset(n_samples=500, seed=0)
    >>> df.shape
    (500, 6)
    """
    if not (0.05 <= proporcao_positivos <= 0.95):
        raise ValueError(
            f"proporcao_positivos deve estar entre 0.05 e 0.95, "
            f"recebido: {proporcao_positivos}"
        )

    rng = np.random.default_rng(seed)

    # Gera vetor de fraude (target) com a proporção especificada
    fraude = rng.choice(
        [0, 1],
        size=n_samples,
        p=[1 - proporcao_positivos, proporcao_positivos],
    )

    # Valor da transação: fraudes tendem a ser de alto valor
    valor_transacao = np.where(
        fraude,
        rng.uniform(500, 10_000, n_samples),  # fraude: R$500 – R$10.000
        rng.uniform(10, 800, n_samples),  # legítimo: R$10 – R$800
    ).round(2)

    # Hora da transação: fraudes ocorrem mais de madrugada
    hora_transacao = np.where(
        fraude,
        rng.integers(0, 6, n_samples),  # fraude: 0h–5h
        rng.integers(7, 23, n_samples),  # legítimo: 7h–22h
    )

    # Distância da última compra (km): fraude costuma ter distância grande
    distancia_ultima_compra = np.where(
        fraude,
        rng.uniform(100, 5_000, n_samples),  # fraude: 100–5000 km
        rng.uniform(0, 50, n_samples),  # legítimo: 0–50 km
    ).round(1)

    # Tentativas de senha: ataque de força bruta → muitas tentativas
    tentativas_senha = np.where(
        fraude,
        rng.integers(2, 10, n_samples),  # fraude: 2–9 tentativas
        rng.integers(1, 2, n_samples),  # legítimo: quase sempre 1
    )

    # País diferente do habitual: 40% das fraudes vs 5% dos legítimos
    pais_diferente = (rng.random(n_samples) < np.where(fraude, 0.40, 0.05)).astype(int)

    df = pd.DataFrame(
        {
            "valor_transacao": valor_transacao,
            "hora_transacao": hora_transacao,
            "distancia_ultima_compra": distancia_ultima_compra,
            "tentativas_senha": tentativas_senha,
            "pais_diferente": pais_diferente,
            "target": fraude,
        }
    )

    X = df.drop(columns=["target"]).values
    y = df["target"].values
    return df, X, y


# ---------------------------------------------------------------------------
# Script principal — demonstração e validação dos checkpoints
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Exercício 1.1 — make_classification com class_sep variado ===")
    for sep in [0.5, 1.0, 3.0]:
        X_ref, y_ref = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            class_sep=sep,
            weights=[0.7, 0.3],
            random_state=42,
        )
        df_ref = pd.DataFrame(X_ref, columns=[f"feature_{i}" for i in range(5)])
        df_ref["target"] = y_ref
        print(
            f"\nclass_sep={sep} | Distribuição: {df_ref['target'].value_counts().to_dict()}"
        )
        # Observação: class_sep=0.5 → classes se misturam mais (problema mais difícil)
        #             class_sep=3.0 → classes muito separadas (problema muito fácil, inútil)
        #             A distribuição do target não muda; o que muda é a separabilidade das classes.

    print("\n=== Exercício 1.2 + 1.3 — gerar_dataset com domínio fraude ===")
    df, X, y = gerar_dataset(n_samples=2000, seed=42)
    print(f"Shape: {df.shape}")
    print(f"\nDistribuição do target:\n{df['target'].value_counts()}")
    print(f"\nMédias por classe:\n{df.groupby('target').mean().round(2)}")

    print("\n=== Checkpoint 1.3 — Validação do proporcao_positivos ===")
    try:
        gerar_dataset(proporcao_positivos=1.5)
    except ValueError as e:
        print(f"ValueError capturado corretamente: {e}")

    df2, X2, y2 = gerar_dataset(n_samples=500, seed=0)
    print(
        f"\nRetorno (df, X, y): df.shape={df2.shape}, X.shape={X2.shape}, y.shape={y2.shape}"
    )
