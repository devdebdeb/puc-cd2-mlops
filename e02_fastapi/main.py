"""
e02 — FastAPI na Prática: Bella Tavola 🍝
==========================================
Matéria : CDIA CD2 2026
Conceitos: Roteamento, path/query params, modelos Pydantic, response_model,
           HTTPException, Field validators, CRUD completo (pratos, bebidas, pedidos)

Estrutura do arquivo
--------------------
- Bloco 1 (ex 1.1-1.8) : GET/POST básicos, filtros, modelos de entrada e saída
- Bloco 2 (ex 2.1-2.5) : HTTPException, Field, @field_validator, PUT
- Bloco 3 (ex 3.1-3.2) : POST /pedidos com regra de negócio e lógica de valor total
"""

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bella Tavola API",
    description="API do restaurante Bella Tavola - Exercício e02 CDIA CD2 2026",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# BLOCO 1 - Exercício 1.1
# Rota raiz com campos extras (chef, cidade, especialidade)
# Conceito: GET simples, retorno de dicionário
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


# ---------------------------------------------------------------------------
# Dados em memória (estado global da aplicação)
# Exercícios 1.2 - 2.x atualizam progressivamente esta lista
# ---------------------------------------------------------------------------

# Exercício 1.5: campo 'disponivel' adicionado conforme gabarito
pratos = [
    {"id": 1, "nome": "Margherita",      "categoria": "pizza",     "preco": 45.0,  "disponivel": True,  "descricao": None, "criado_em": "2024-01-01T00:00:00"},
    {"id": 2, "nome": "Carbonara",       "categoria": "massa",     "preco": 52.0,  "disponivel": True,  "descricao": None, "criado_em": "2024-01-01T00:00:00"},
    {"id": 3, "nome": "Lasanha Bolonhesa","categoria": "massa",    "preco": 58.0,  "disponivel": False, "descricao": None, "criado_em": "2024-01-01T00:00:00"},
    {"id": 4, "nome": "Tiramisù",        "categoria": "sobremesa", "preco": 28.0,  "disponivel": True,  "descricao": None, "criado_em": "2024-01-01T00:00:00"},
    {"id": 5, "nome": "Quattro Stagioni","categoria": "pizza",     "preco": 49.0,  "disponivel": True,  "descricao": None, "criado_em": "2024-01-01T00:00:00"},
    {"id": 6, "nome": "Panna Cotta",     "categoria": "sobremesa", "preco": 24.0,  "disponivel": True,  "descricao": None, "criado_em": "2024-01-01T00:00:00"},
]

bebidas = [
    {"id": 1, "nome": "Água Mineral",    "tipo": "agua",        "preco": 8.0,   "alcoolica": False, "volume_ml": 500, "criado_em": "2024-01-01T00:00:00"},
    {"id": 2, "nome": "Chianti Classico","tipo": "vinho",       "preco": 120.0, "alcoolica": True,  "volume_ml": 750, "criado_em": "2024-01-01T00:00:00"},
    {"id": 3, "nome": "San Pellegrino",  "tipo": "agua",        "preco": 15.0,  "alcoolica": False, "volume_ml": 750, "criado_em": "2024-01-01T00:00:00"},
    {"id": 4, "nome": "Suco de Laranja", "tipo": "suco",        "preco": 18.0,  "alcoolica": False, "volume_ml": 300, "criado_em": "2024-01-01T00:00:00"},
    {"id": 5, "nome": "Prosecco",        "tipo": "vinho",       "preco": 95.0,  "alcoolica": True,  "volume_ml": 750, "criado_em": "2024-01-01T00:00:00"},
]

pedidos = []


# ---------------------------------------------------------------------------
# Modelos Pydantic — Bloco 2 (ex 2.3): Field com constraints
# ---------------------------------------------------------------------------

class PratoInput(BaseModel):
    """Modelo de entrada para criação de prato.

    Conceitos aplicados:
    - Field(min_length, max_length) para strings
    - Field(gt=0) para preço positivo
    - Field(pattern=...) para categoria enumerada via regex
    - Optional com default
    - bool com default
    """
    nome: str = Field(min_length=3, max_length=100, description="Nome do prato")
    categoria: str = Field(
        pattern=r"^(pizza|massa|sobremesa|entrada|salada)$",
        description="Categoria: pizza, massa, sobremesa, entrada ou salada",
    )
    preco: float = Field(gt=0, description="Preço em reais, deve ser positivo")
    descricao: Optional[str] = Field(default=None, max_length=500)
    disponivel: bool = True

    # Bloco 2, ex 2.4: validador de regra de negócio
    @field_validator("nome")
    @classmethod
    def nome_nao_pode_ser_so_espacos(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("O nome do prato não pode conter apenas espaços")
        return v.strip()


class PratoOutput(BaseModel):
    """Modelo de saída — inclui id e criado_em gerados pelo servidor."""
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
    """Bloco 3: pedido vinculado a um prato existente e disponível."""
    prato_id: int
    quantidade: int = Field(ge=1, description="Quantidade mínima de 1")
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
# BLOCO 1 — Exercícios 1.2 a 1.7: Rotas de Pratos
# ---------------------------------------------------------------------------

@app.get("/pratos")
async def listar_pratos(
    categoria: Optional[str] = None,
    preco_maximo: Optional[float] = None,
    apenas_disponiveis: bool = False,
):
    """
    Exercício 1.4 + 1.5: GET /pratos com dois query params opcionais
    - categoria (str)
    - preco_maximo (float)
    - apenas_disponiveis (bool, default False)
    """
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
    """
    Exercício 1.3 + 1.5 + 2.2:
    - path parameter prato_id
    - query parameter formato (completo/resumido)
    - HTTPException 404 quando não encontrado
    """
    for prato in pratos:
        if prato["id"] == prato_id:
            if formato == "resumido":
                return {"nome": prato["nome"], "preco": prato["preco"]}
            return prato
    raise HTTPException(
        status_code=404,
        detail=f"Prato com id {prato_id} não encontrado",
    )


@app.post("/pratos", response_model=PratoOutput, status_code=201)
async def criar_prato(prato: PratoInput):
    """
    Exercício 1.6 + 1.7: POST com PratoInput, retorna PratoOutput
    - gera id automaticamente
    - adiciona criado_em pelo servidor
    """
    novo_id = max(p["id"] for p in pratos) + 1 if pratos else 1
    novo_prato = {
        "id": novo_id,
        "criado_em": datetime.now().isoformat(),
        **prato.model_dump(),
    }
    pratos.append(novo_prato)
    return novo_prato


# Exercício 2.5: PUT para alterar disponibilidade
@app.put("/pratos/{prato_id}/disponibilidade")
async def atualizar_disponibilidade(prato_id: int, body: DisponibilidadeInput):
    """Bloco 2 — atualiza campo 'disponivel' de um prato existente."""
    for prato in pratos:
        if prato["id"] == prato_id:
            prato["disponivel"] = body.disponivel
            return prato
    raise HTTPException(status_code=404, detail=f"Prato com id {prato_id} não encontrado")


# ---------------------------------------------------------------------------
# BLOCO 1 — Exercício 1.8: Rotas de Bebidas (desafio autônomo)
# ---------------------------------------------------------------------------

@app.get("/bebidas")
async def listar_bebidas(
    tipo: Optional[str] = None,
    alcoolica: Optional[bool] = None,
):
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
        status_code=404,
        detail=f"Bebida com id {bebida_id} não encontrada",
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


# ---------------------------------------------------------------------------
# BLOCO 3 — Pedidos com regra de negócio
# ---------------------------------------------------------------------------

@app.post("/pedidos", response_model=PedidoOutput, status_code=201)
async def criar_pedido(pedido: PedidoInput):
    """
    Regras:
    - prato_id deve existir → 404 se não existir
    - prato deve estar disponível → 400 se indisponível
    - valor_total = preco * quantidade
    """
    prato = next((p for p in pratos if p["id"] == pedido.prato_id), None)

    if prato is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prato com id {pedido.prato_id} não encontrado",
        )

    if not prato["disponivel"]:
        raise HTTPException(
            status_code=400,
            detail=f"O prato '{prato['nome']}' não está disponível no momento",
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
