"""
e03 — Bloco 2: Treinar e serializar o modelo
=============================================
Matéria  : CDIA CD2 2026
Conceitos: train_test_split, RandomForestClassifier, classification_report,
           métricas (precision, recall, F1), joblib.dump, ciclo save→load→predict

Exercícios cobertos:
  2.1 — Treinar e avaliar (RandomForest + classification_report)
  2.2 — Serializar o artefato (joblib.dump, ciclo load, assert predições iguais)
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from gerar_dados import gerar_dataset


# ---------------------------------------------------------------------------
# Exercício 2.1 — Treinar e avaliar
# ---------------------------------------------------------------------------
def treinar_e_avaliar(n_samples: int = 2000, seed: int = 42) -> RandomForestClassifier:
    df, X, y = gerar_dataset(n_samples=n_samples, seed=seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,  # mantém proporção de classes no split
    )

    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("=== Relatório de Classificação ===")
    print(classification_report(y_test, y_pred, target_names=["legítimo", "fraude"]))

    # Reflexão pedida pelo exercício:
    # - Para o domínio de fraude, RECALL é mais crítico que PRECISION.
    #   Deixar passar uma fraude (falso negativo) costuma ser mais custoso
    #   do que bloquear uma transação legítima (falso positivo).
    # - Um recall alto para a classe 'fraude' significa que o modelo detecta
    #   a maioria das fraudes reais — objetivo principal neste domínio.

    return model, X_test, y_test


# ---------------------------------------------------------------------------
# Exercício 2.2 — Serializar o artefato
# ---------------------------------------------------------------------------
def salvar_e_validar(
    model: RandomForestClassifier, X_test: np.ndarray, caminho: str = "model.pkl"
) -> None:
    """
    Ciclo completo: salvar → carregar → predizer → comparar.
    Garante que o artefato é autossuficiente e reproduzível.
    """
    joblib.dump(model, caminho)
    print(f"\nModelo salvo em: {caminho}")

    model_carregado = joblib.load(caminho)

    amostra = X_test[:5]
    pred_original = model.predict(amostra)
    pred_carregado = model_carregado.predict(amostra)

    assert np.array_equal(
        pred_original, pred_carregado
    ), "❌ Predições divergem! O artefato não reproduziu corretamente."
    print(f"✅ Artefato validado — predições idênticas: {pred_original}")


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model, X_test, y_test = treinar_e_avaliar()
    salvar_e_validar(model, X_test)
