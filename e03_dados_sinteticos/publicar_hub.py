import os
import sklearn
import joblib as jl
import numpy as np
from huggingface_hub import HfApi, login

def publicar():
    # Tenta pegar o token do ambiente
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ ERRO: Defina a variável de ambiente HF_TOKEN primeiro.")
        print("Use: $env:HF_TOKEN = 'seu_token'")
        return

    try:
        login(token=token)
        api = HfApi()
        username = api.whoami()["name"]
        repo_id = f"{username}/mlops-fraud-v2"
        
        print(f"🚀 Verificando repositório: {repo_id}...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        # Gerar requirements para o Hub
        with open("requirements_hub.txt", "w") as f:
            f.write(f"scikit-learn=={sklearn.__version__}\njoblib=={jl.__version__}\nnumpy=={np.__version__}\n")

        # Gerar README para o Hub
        model_card = f"""---
language: pt
tags:
  - sklearn
  - classification
  - fraud-detection
  - mlops
---
# Bella Tavola - Fraud Detector v1
Modelo de detecção de fraude (PUC CD2 2026).
"""
        with open("README_hub.md", "w", encoding="utf-8") as f:
            f.write(model_card)

        # Upload
        for local, remoto in {"model.pkl": "model.pkl", "README_hub.md": "README.md", "requirements_hub.txt": "requirements.txt"}.items():
            print(f"⬆️ Enviando {local}...")
            api.upload_file(path_or_fileobj=local, path_in_repo=remoto, repo_id=repo_id, repo_type="model")

        print(f"\n✅ SUCESSO! Modelo em: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ ERRO durante a publicação: {e}")

if __name__ == "__main__":
    publicar()
