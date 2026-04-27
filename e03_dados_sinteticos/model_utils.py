"""
e03 — Bloco 3: Carregar modelo do Hugging Face Hub
===================================================
Matéria  : CDIA CD2 2026
Conceitos: huggingface_hub, hf_hub_download, login via env var,
           separação entre código de treino e código de inferência,
           força de download para invalidar cache

Nota sobre segurança:
  O token HF_TOKEN deve vir de variável de ambiente ou secret do CI.
  NUNCA commitar o valor do token em código ou arquivos de config.
"""

import os

import joblib
from huggingface_hub import hf_hub_download, login


def load_model(
    repo_id: str,
    filename: str = "model.pkl",
    force_download: bool = False,
):
    """
    Carrega o modelo de detecção de fraude publicado no Hugging Face Hub.

    Parâmetros
    ----------
    repo_id : str
        Identificador do repositório no Hub (ex: "usuario/mlops-bella-tavola-v1").
    filename : str
        Nome do arquivo serializado dentro do repositório.
    force_download : bool
        Se True, ignora o cache local e força o download da versão mais recente.

    Retorna
    -------
    modelo : sklearn estimator
        Objeto carregado com joblib, pronto para .predict() e .predict_proba().

    Exemplo de uso
    --------------
    >>> import os
    >>> os.environ["HF_TOKEN"] = "hf_..."
    >>> from model_utils import load_model
    >>> model = load_model("usuario/meu-repo")
    >>> model.predict([[150.0, 2, 500.0, 5, 1]])
    """
    # Autentica com o Hub se o token estiver disponível como variável de ambiente.
    # Em CI, o secret é injetado no step do workflow como env var.
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=force_download,
    )
    return joblib.load(local_path)


if __name__ == "__main__":
    # Demonstração — substitua pelo seu repo_id real após publicar o modelo
    import sys

    repo_id = sys.argv[1] if len(sys.argv) > 1 else "SEU_USUARIO/SEU_REPO"
    print(f"Tentando carregar modelo de: {repo_id}")
    model = load_model(repo_id)
    print(f"Modelo carregado: {type(model)}")
    print(f"Possui predict:       {hasattr(model, 'predict')}")
    print(f"Possui predict_proba: {hasattr(model, 'predict_proba')}")
