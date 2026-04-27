"""
e04 — model_utils.py (idêntico ao e03, reutilizado aqui para o CI)
"""

import os

import joblib
from huggingface_hub import hf_hub_download, login


def load_model(
    repo_id: str,
    filename: str = "model.pkl",
    force_download: bool = False,
):
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=force_download,
    )
    return joblib.load(local_path)
