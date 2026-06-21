# e05 — Dockerfile da API Bella Tavola
# =====================================
# Containeriza o app FastAPI do e04_ci (rota /ml/predict + modelo do HF Hub).
#
# Decisões (e05_p01 e e05_p02):
#   - Multi-stage: o estágio 'builder' instala as dependências num venv isolado;
#     o estágio 'runtime' copia só o venv pronto — imagem final menor e mais limpa.
#   - Usuário não-root (appuser): o processo não roda como root dentro do contêiner.
#   - --host 0.0.0.0: o uvicorn precisa ouvir em todas as interfaces para que o
#     -p do docker run consiga rotear o tráfego do host até o contêiner.
#
# O modelo NÃO entra na imagem: é baixado do Hugging Face Hub em runtime,
# usando as variáveis de ambiente HF_TOKEN e HF_REPO_ID (carregamento lazy).

# ---------- Estágio 1: builder ----------
FROM python:3.11-slim AS builder

WORKDIR /app

# Cria um virtualenv isolado para as dependências
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instala as dependências primeiro (layer cacheável: só refaz se requirements mudar)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- Estágio 2: runtime ----------
FROM python:3.11-slim AS runtime

# Usuário não-root para rodar a aplicação
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Copia o venv já instalado do estágio builder (sem trazer o cache do pip)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copia o código da aplicação já com dono appuser
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# 0.0.0.0 é essencial: sem isso o uvicorn ouviria só em 127.0.0.1 dentro do
# contêiner e o -p 8000:8000 não conseguiria alcançá-lo a partir do host.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
