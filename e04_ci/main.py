"""
e04 — main.py: API Bella Tavola com rota /ml/predict
=====================================================
Reutiliza a API do e02 e adiciona o endpoint de predição do modelo.
O modelo é carregado do Hugging Face Hub via model_utils.load_model.
"""

import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from routers.predict import router as predict_router

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bella Tavola API",
    description="API do Bella Tavola com CI/CD — e04 CDIA CD2 2026",
    version="2.0.0",
)

# Inclui o router de ML
app.include_router(predict_router, prefix="/ml", tags=["ML"])


# ---------------------------------------------------------------------------
# Dados em memória
# ---------------------------------------------------------------------------
pratos = [
    {
        "id": 1,
        "nome": "Margherita",
        "categoria": "pizza",
        "preco": 45.0,
        "disponivel": True,
        "descricao": None,
        "criado_em": "2024-01-01T00:00:00",
    },
    {
        "id": 2,
        "nome": "Carbonara",
        "categoria": "massa",
        "preco": 52.0,
        "disponivel": True,
        "descricao": None,
        "criado_em": "2024-01-01T00:00:00",
    },
    {
        "id": 3,
        "nome": "Lasanha Bolonhesa",
        "categoria": "massa",
        "preco": 58.0,
        "disponivel": False,
        "descricao": None,
        "criado_em": "2024-01-01T00:00:00",
    },
    {
        "id": 4,
        "nome": "Tiramisù",
        "categoria": "sobremesa",
        "preco": 28.0,
        "disponivel": True,
        "descricao": None,
        "criado_em": "2024-01-01T00:00:00",
    },
    {
        "id": 5,
        "nome": "Quattro Stagioni",
        "categoria": "pizza",
        "preco": 49.0,
        "disponivel": True,
        "descricao": None,
        "criado_em": "2024-01-01T00:00:00",
    },
    {
        "id": 6,
        "nome": "Panna Cotta",
        "categoria": "sobremesa",
        "preco": 24.0,
        "disponivel": True,
        "descricao": None,
        "criado_em": "2024-01-01T00:00:00",
    },
]

bebidas = [
    {
        "id": 1,
        "nome": "Água Mineral",
        "tipo": "agua",
        "preco": 8.0,
        "alcoolica": False,
        "volume_ml": 500,
        "criado_em": "2024-01-01T00:00:00",
    },
    {
        "id": 2,
        "nome": "Chianti Classico",
        "tipo": "vinho",
        "preco": 120.0,
        "alcoolica": True,
        "volume_ml": 750,
        "criado_em": "2024-01-01T00:00:00",
    },
    {
        "id": 3,
        "nome": "San Pellegrino",
        "tipo": "agua",
        "preco": 15.0,
        "alcoolica": False,
        "volume_ml": 750,
        "criado_em": "2024-01-01T00:00:00",
    },
    {
        "id": 4,
        "nome": "Suco de Laranja",
        "tipo": "suco",
        "preco": 18.0,
        "alcoolica": False,
        "volume_ml": 300,
        "criado_em": "2024-01-01T00:00:00",
    },
    {
        "id": 5,
        "nome": "Prosecco",
        "tipo": "vinho",
        "preco": 95.0,
        "alcoolica": True,
        "volume_ml": 750,
        "criado_em": "2024-01-01T00:00:00",
    },
]

pedidos = []


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------
class PratoInput(BaseModel):
    nome: str = Field(min_length=3, max_length=100)
    categoria: str = Field(pattern=r"^(pizza|massa|sobremesa|entrada|salada)$")
    preco: float = Field(gt=0)
    descricao: Optional[str] = Field(default=None, max_length=500)
    disponivel: bool = True

    @field_validator("nome")
    @classmethod
    def nome_nao_pode_ser_so_espacos(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("O nome do prato não pode conter apenas espaços")
        return v.strip()


class PratoOutput(BaseModel):
    id: int
    nome: str
    categoria: str
    preco: float
    descricao: Optional[str]
    disponivel: bool
    criado_em: str


class BebidaInput(BaseModel):
    nome: str = Field(min_length=3, max_length=100)
    tipo: str = Field(pattern=r"^(vinho|agua|refrigerante|suco|cerveja)$")
    preco: float = Field(gt=0)
    alcoolica: bool
    volume_ml: int = Field(ge=50, le=2000)


class BebidaOutput(BaseModel):
    id: int
    nome: str
    tipo: str
    preco: float
    alcoolica: bool
    volume_ml: int
    criado_em: str


class PedidoInput(BaseModel):
    prato_id: int
    quantidade: int = Field(ge=1)
    observacao: Optional[str] = Field(default=None, max_length=300)


class PedidoOutput(BaseModel):
    id: int
    prato_id: int
    nome_prato: str
    quantidade: int
    preco_unitario: float
    valor_total: float
    observacao: Optional[str]
    criado_em: str


class DisponibilidadeInput(BaseModel):
    disponivel: bool


# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "restaurante": "Bella Tavola",
        "mensagem": "Bem-vindo à nossa API",
        "chef": "Marco Rossi",
        "cidade": "São Paulo",
        "especialidade": "Massas artesanais",
    }


@app.get("/health/db")
async def health_db():
    """
    Verifica a conexão com o PostgreSQL (serviço 'db' do docker-compose).

    Demonstra o conceito do e05_p02 (Exercício 11.2): a API alcança o banco
    pelo NOME DO SERVIÇO na rede interna do Compose, via DATABASE_URL.
    Sem DATABASE_URL (ex: nos testes) retorna 503 — não quebra a suíte.
    """
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise HTTPException(status_code=503, detail="DATABASE_URL não configurada")
    try:
        import psycopg2

        conn = psycopg2.connect(database_url, connect_timeout=3)
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            versao = cur.fetchone()[0]
        conn.close()
        return {"status": "ok", "database": "postgresql", "versao": versao}
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Falha ao conectar no banco: {exc}"
        )


@app.get("/pratos")
async def listar_pratos(
    categoria: Optional[str] = None,
    preco_maximo: Optional[float] = None,
    apenas_disponiveis: bool = False,
):
    resultado = pratos
    if categoria:
        resultado = [p for p in resultado if p["categoria"] == categoria]
    if preco_maximo is not None:
        resultado = [p for p in resultado if p["preco"] <= preco_maximo]
    if apenas_disponiveis:
        resultado = [p for p in resultado if p["disponivel"]]
    return resultado


@app.get("/pratos/{prato_id}")
async def buscar_prato(prato_id: int, formato: str = "completo"):
    for prato in pratos:
        if prato["id"] == prato_id:
            if formato == "resumido":
                return {"nome": prato["nome"], "preco": prato["preco"]}
            return prato
    raise HTTPException(
        status_code=404, detail=f"Prato com id {prato_id} não encontrado"
    )


@app.post("/pratos", response_model=PratoOutput, status_code=201)
async def criar_prato(prato: PratoInput):
    novo_id = max(p["id"] for p in pratos) + 1 if pratos else 1
    novo_prato = {
        "id": novo_id,
        "criado_em": datetime.now().isoformat(),
        **prato.model_dump(),
    }
    pratos.append(novo_prato)
    return novo_prato


@app.put("/pratos/{prato_id}/disponibilidade")
async def atualizar_disponibilidade(prato_id: int, body: DisponibilidadeInput):
    for prato in pratos:
        if prato["id"] == prato_id:
            prato["disponivel"] = body.disponivel
            return prato
    raise HTTPException(
        status_code=404, detail=f"Prato com id {prato_id} não encontrado"
    )


@app.get("/bebidas")
async def listar_bebidas(tipo: Optional[str] = None, alcoolica: Optional[bool] = None):
    resultado = bebidas
    if tipo:
        resultado = [b for b in resultado if b["tipo"] == tipo]
    if alcoolica is not None:
        resultado = [b for b in resultado if b["alcoolica"] == alcoolica]
    return resultado


@app.get("/bebidas/{bebida_id}")
async def buscar_bebida(bebida_id: int):
    for bebida in bebidas:
        if bebida["id"] == bebida_id:
            return bebida
    raise HTTPException(
        status_code=404, detail=f"Bebida com id {bebida_id} não encontrada"
    )


@app.post("/bebidas", response_model=BebidaOutput, status_code=201)
async def criar_bebida(bebida: BebidaInput):
    novo_id = max(b["id"] for b in bebidas) + 1 if bebidas else 1
    nova_bebida = {
        "id": novo_id,
        "criado_em": datetime.now().isoformat(),
        **bebida.model_dump(),
    }
    bebidas.append(nova_bebida)
    return nova_bebida


@app.post("/pedidos", response_model=PedidoOutput, status_code=201)
async def criar_pedido(pedido: PedidoInput):
    prato = next((p for p in pratos if p["id"] == pedido.prato_id), None)
    if prato is None:
        raise HTTPException(
            status_code=404, detail=f"Prato com id {pedido.prato_id} não encontrado"
        )
    if not prato["disponivel"]:
        raise HTTPException(
            status_code=400, detail=f"O prato '{prato['nome']}' não está disponível"
        )
    novo_id = max(p["id"] for p in pedidos) + 1 if pedidos else 1
    novo_pedido = {
        "id": novo_id,
        "prato_id": pedido.prato_id,
        "nome_prato": prato["nome"],
        "quantidade": pedido.quantidade,
        "preco_unitario": prato["preco"],
        "valor_total": round(prato["preco"] * pedido.quantidade, 2),
        "observacao": pedido.observacao,
        "criado_em": datetime.now().isoformat(),
    }
    pedidos.append(novo_pedido)
    return novo_pedido
