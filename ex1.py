import json
import os
import re
import sqlite3
from datetime import date, datetime, timedelta
from typing import TypedDict
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# ============================================================
# CONFIGURAÇÃO
# ============================================================

# Modos disponíveis:
#
# MODO_TESTE = True
#   Usa regras determinísticas.
#   Não utiliza nenhuma LLM.
#
# MODO_LOCAL = True
#   Usa Qwen3 1.7B através do Ollama.
#
# Se os dois forem False:
#   Usa Qwen3 4B através da Hugging Face.

MODO_TESTE = False
MODO_LOCAL = True

# ============================================================
# CONFIGURAÇÃO DO HUGGING FACE
# ============================================================

load_dotenv()

token = os.getenv("token_teste")

client = None

if token:
    client = InferenceClient(api_key=token)


# ============================================================
# CONFIGURAÇÃO DO OLLAMA
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODELO = "qwen3:1.7b"


def chamar_ollama(system_prompt: str, user_message: str):
    """
    Envia uma mensagem para o modelo local através da API
    HTTP do Ollama.

    O retorno é apenas o conteúdo textual produzido pelo modelo.
    """

    payload = {
        "model": OLLAMA_MODELO,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
    }

    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(request, timeout=60) as response:
        dados = json.load(response)

    return dados["message"]["content"].strip()


# ============================================================
# EVENTOS ARMAZENADOS NO SQLITE
# ============================================================

class Evento(TypedDict):
    id: int
    nome: str
    data: str
    horario: str


BANCO_EVENTOS = "eventos.db"


def conectar_eventos():
    return sqlite3.connect(BANCO_EVENTOS)


def inicializar_banco_eventos():
    conn = conectar_eventos()

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS eventos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            data TEXT NOT NULL,
            horario TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()

def linha_para_evento(linha) -> Evento:
    return {
        "id": linha[0],
        "nome": linha[1],
        "data": linha[2],
        "horario": linha[3],
    }

def listar_eventos() -> list[Evento]:
    conn = conectar_eventos()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, nome, data, horario
        FROM eventos
        ORDER BY data, horario
        """
    )

    linhas = cursor.fetchall()

    conn.close()

    return [
        linha_para_evento(linha)
        for linha in linhas
    ]

def inserir_eventos_iniciais():
    conn = conectar_eventos()

    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM eventos"
    )

    quantidade = cursor.fetchone()[0]

    if quantidade == 0:
        hoje = date.today()

        eventos_iniciais = [
            (
                "Palestra de Inteligência Artificial",
                (hoje + timedelta(days=1)).isoformat(),
                "19:00",
            ),
            (
                "Reunião do projeto TP3",
                (hoje + timedelta(days=2)).isoformat(),
                "14:00",
            ),
            (
                "Entrega do relatório",
                (hoje + timedelta(days=4)).isoformat(),
                "23:59",
            ),
        ]

        cursor.executemany(
            """
            INSERT INTO eventos
            (nome, data, horario)
            VALUES (?, ?, ?)
            """,
            eventos_iniciais,
        )

        conn.commit()

    conn.close()


# ============================================================
# ESTADO COMPARTILHADO DO GRAFO
# ============================================================

class Estado(TypedDict):
    mensagem: str
    intencao: str
    nome_evento: str
    data_evento: str
    horario_evento: str
    eventos: list[Evento]
    tentativas: int
    resposta: str
    data_normalizada: str
    busca_alternativa: bool
    erro_llm: str


# ============================================================
# NÓ: NORMALIZAÇÃO DA DATA
# ============================================================

def normalizar_data(state: Estado):
    print("\n[normalizar_data] Interpretando a data...")

    mensagem = state["mensagem"].lower()
    hoje = date.today()
    data_normalizada = ""

    if (
        "depois de amanhã" in mensagem
        or "depois de amanha" in mensagem
    ):
        data_normalizada = (
            hoje + timedelta(days=2)
        ).isoformat()

    elif re.search(
        r"\bamanhã\b|\bamanha\b",
        mensagem,
    ):
        data_normalizada = (
            hoje + timedelta(days=1)
        ).isoformat()

    elif re.search(
        r"\bhoje\b",
        mensagem,
    ):
        data_normalizada = hoje.isoformat()

    else:
        match_iso = re.search(
            r"\b\d{4}-\d{2}-\d{2}\b",
            mensagem,
        )

        match_br = re.search(
            r"\b\d{2}/\d{2}/\d{4}\b",
            mensagem,
        )

        try:
            if match_iso:
                data_normalizada = date.fromisoformat(
                    match_iso.group()
                ).isoformat()

            elif match_br:
                data_normalizada = datetime.strptime(
                    match_br.group(),
                    "%d/%m/%Y",
                ).date().isoformat()

        except ValueError:
            data_normalizada = ""

    print(
        f"[normalizar_data] "
        f"Data normalizada: "
        f"{data_normalizada or 'não informada'}"
    )

    return {
        "data_normalizada": data_normalizada,
    }


# ============================================================
# NÓ: NORMALIZAÇÃO DO HORÁRIO
# ============================================================

def normalizar_horario(state: Estado):
    print("\n[normalizar_horario] Interpretando o horário...")

    horario = state["horario_evento"].strip().lower()

    horario_normalizado = ""

    match = re.search(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(?:h|hs|horas)?\b",
        horario,
    )

    if match:
        hora = int(match.group(1))
        minuto = int(match.group(2) or 0)

        if 0 <= hora <= 23 and 0 <= minuto <= 59:
            horario_normalizado = f"{hora:02d}:{minuto:02d}"

    print(
        f"[normalizar_horario] "
        f"Horário normalizado: "
        f"{horario_normalizado or 'não informado'}"
    )

    return {
        "horario_evento": horario_normalizado,
    }


# ============================================================
# NÓ: ANÁLISE DA INTENÇÃO COM OLLAMA
# ============================================================

def analisar_intencao_ollama(state: Estado):
    print(
        "\n[analisar_intencao] "
        "Interpretando mensagem com Ollama..."
    )

    system_prompt = """
Você classifica intenções de um sistema de gerenciamento de eventos.

Existem somente quatro intenções:

criar:
quando o usuário quer criar, cadastrar, adicionar ou marcar
um novo evento.

consultar:
quando o usuário quer saber informações sobre eventos
existentes ou listar eventos.

cancelar:
quando o usuário quer cancelar ou remover um evento existente.

desconhecida:
quando a mensagem não representa nenhuma operação de
gerenciamento de eventos.

Exemplos:

"Quero cadastrar uma palestra amanhã"
-> criar

"Crie uma reunião para sexta"
-> criar

"Quando é a palestra de inteligência artificial?"
-> consultar

"Quais eventos tenho amanhã?"
-> consultar

"Cancele a palestra de inteligência artificial"
-> cancelar

"Remova a reunião de amanhã"
-> cancelar

"Qual é a capital da França?"
-> desconhecida

"Me explique como funciona Python"
-> desconhecida

Responda SOMENTE com uma destas quatro palavras:

criar
consultar
cancelar
desconhecida

Não explique.
Não faça perguntas.
Não escreva nenhuma outra palavra.
"""

    try:
        resposta = chamar_ollama(
            system_prompt,
            state["mensagem"],
        )

        intencao = resposta.strip().lower()

        intencoes_validas = {
            "consultar",
            "criar",
            "cancelar",
            "desconhecida",
        }

        if intencao not in intencoes_validas:
            intencao = "desconhecida"

        print(
            f"[analisar_intencao] "
            f"Intenção identificada: {intencao}"
        )

        return {
            "intencao": intencao,
            "erro_llm": "",
        }

    except (HTTPError, URLError, TimeoutError) as erro:
        print(
            "[analisar_intencao] "
            f"Erro ao acessar Ollama: {erro}"
        )

        return {
            "erro_llm": (
                "Não foi possível acessar o modelo local "
                "através do Ollama."
            )
        }

    except Exception as erro:
        print(
            "[analisar_intencao] "
            f"Erro inesperado na LLM local: {erro}"
        )

        return {
            "erro_llm": (
                "Ocorreu um erro ao processar "
                "a mensagem com a inteligência artificial."
            )
        }


# ============================================================
# NÓ: ANÁLISE DA INTENÇÃO COM HUGGING FACE
# ============================================================

def analisar_intencao_llm(state: Estado):
    print(
        "\n[analisar_intencao] "
        "Interpretando mensagem com Hugging Face..."
    )

    try:
        if client is None:
            raise RuntimeError(
                "Token da Hugging Face não configurado."
            )

        resposta = client.chat.completions.create(
            model="Qwen/Qwen3-4B-Instruct-2507",
            messages=[
                {
                    "role": "system",
                    "content": """
Você classifica intenções em um sistema de gerenciamento
de eventos.

As intenções possíveis são:

- consultar: buscar ou listar eventos existentes.
- criar: cadastrar ou adicionar um evento.
- cancelar: cancelar ou remover um evento.
- desconhecida: qualquer outra intenção.

Exemplos:

"Quero cadastrar uma palestra amanhã" -> criar
"Quando é a palestra?" -> consultar
"Cancele a palestra" -> cancelar
"Qual é a capital da França?" -> desconhecida

Responda exclusivamente com uma destas palavras:

consultar
criar
cancelar
desconhecida

Não faça perguntas.
Não explique sua resposta.
"""
                },
                {
                    "role": "user",
                    "content": state["mensagem"],
                },
            ],
            max_tokens=10,
        )

        intencao = (
            resposta.choices[0].message.content
            .strip()
            .lower()
        )

        intencoes_validas = {
            "consultar",
            "criar",
            "cancelar",
            "desconhecida",
        }

        if intencao not in intencoes_validas:
            intencao = "desconhecida"

        print(
            f"[analisar_intencao] "
            f"Intenção identificada: {intencao}"
        )

        return {
            "intencao": intencao,
            "erro_llm": "",
        }

    except HfHubHTTPError as erro:
        print(
            "[analisar_intencao] "
            f"Erro na API da Hugging Face: {erro}"
        )

        return {
            "erro_llm": (
                "Não foi possível acessar o serviço "
                "de inteligência artificial."
            )
        }

    except Exception as erro:
        print(
            "[analisar_intencao] "
            f"Erro inesperado na LLM: {erro}"
        )

        return {
            "erro_llm": (
                "Ocorreu um erro ao processar "
                "a mensagem com a inteligência artificial."
            )
        }


# ============================================================
# NÓ: ANÁLISE DA INTENÇÃO PARA TESTES
# ============================================================

def analisar_intencao_teste(state: Estado):
    print(
        "\n[analisar_intencao] "
        "Interpretando mensagem (modo teste)..."
    )

    mensagem = state["mensagem"].lower()

    if any(
        palavra in mensagem
        for palavra in [
            "cancelar",
            "cancele",
            "cancela",
            "remover",
            "remova",
        ]
    ):
        intencao = "cancelar"

    elif any(
        palavra in mensagem
        for palavra in [
            "cadastrar",
            "cadastre",
            "criar",
            "crie",
            "adicionar",
            "adicione",
            "marcar",
        ]
    ):
        intencao = "criar"

    elif any(
        palavra in mensagem
        for palavra in [
            "quando",
            "qual",
            "quais",
            "listar",
            "liste",
            "mostrar",
            "mostre",
            "tem",
            "tenho",
        ]
    ):
        intencao = "consultar"

    else:
        intencao = "desconhecida"

    print(
        f"[analisar_intencao] "
        f"Intenção identificada: {intencao}"
    )

    return {
        "intencao": intencao,
        "erro_llm": "",
    }


# ============================================================
# NÓ: EXTRAÇÃO DETERMINÍSTICA
#
# A extração continua sendo feita pelo Python.
#
# Motivo:
# o modelo local demonstrou dificuldade para interpretar
# corretamente datas relativas como "amanhã".
#
# Dessa forma:
#
# LLM -> interpreta intenção
# Python -> extrai e normaliza dados estruturados
# LangGraph -> controla o fluxo
# ============================================================
def extrair_dados_teste(state: Estado):
    print(
        "\n[extrair_dados] "
        "Extraindo informações da mensagem..."
    )

    mensagem = state["mensagem"].lower().strip()

    nome_evento = ""
    data_evento = ""
    horario_evento = ""

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    if "depois de amanhã" in mensagem:
        data_evento = "depois de amanhã"

    elif (
        "amanhã" in mensagem
        or "amanha" in mensagem
    ):
        data_evento = "amanhã"

    elif "hoje" in mensagem:
        data_evento = "hoje"

    else:
        match_iso = re.search(
            r"\b\d{4}-\d{2}-\d{2}\b",
            mensagem,
        )

        match_br = re.search(
            r"\b\d{2}/\d{2}/\d{4}\b",
            mensagem,
        )

        if match_iso:
            data_evento = match_iso.group()

        elif match_br:
            data_evento = match_br.group()

    # --------------------------------------------------------
    # HORÁRIO
    # --------------------------------------------------------

    match_horario = re.search(
        r"\b\d{1,2}(?::\d{2})?\s*(?:h|hs|horas)\b",
        mensagem,
    )

    if match_horario:
        horario_evento = match_horario.group().strip()

    # --------------------------------------------------------
    # NOME DO EVENTO
    # --------------------------------------------------------

    texto_nome = mensagem

    expressoes_remover = [
        # Criação
        "quero cadastrar",
        "quero criar",
        "quero adicionar",
        "quero marcar",
        "gostaria de cadastrar",
        "gostaria de criar",
        "gostaria de adicionar",
        "gostaria de marcar",
        "cadastrar",
        "cadastre",
        "criar",
        "crie",
        "adicionar",
        "adicione",
        "marcar",
        "marque",

        # Cancelamento
        "quero cancelar",
        "gostaria de cancelar",
        "cancele",
        "cancelar",
        "cancela",
        "remover",
        "remova",

        # Consultas
        "quando é",
        "quando sera",
        "quando será",
        "quando vai ser",
        "qual é",
        "qual o",
        "qual a",
        "quais eventos tenho",
        "quais eventos tem",
        "quais eventos estão",
        "quais eventos existem",
        "quais eventos vão acontecer",
        "listar eventos",
        "liste os eventos",
        "mostrar eventos",
        "mostre os eventos",
        "me mostre os eventos",
        "me mostre",
        "me diga",
        "quero saber",
        "gostaria de saber",
    ]

    for trecho in expressoes_remover:
        texto_nome = texto_nome.replace(
            trecho,
            " ",
        )

    # --------------------------------------------------------
    # REFERÊNCIAS TEMPORAIS
    # --------------------------------------------------------

    texto_nome = re.sub(
        r"\bdepois de amanhã\b",
        " ",
        texto_nome,
    )

    texto_nome = re.sub(
        r"\bamanhã\b|\bamanha\b|\bhoje\b",
        " ",
        texto_nome,
    )

    texto_nome = re.sub(
        r"\b\d{4}-\d{2}-\d{2}\b",
        " ",
        texto_nome,
    )

    texto_nome = re.sub(
        r"\b\d{2}/\d{2}/\d{4}\b",
        " ",
        texto_nome,
    )

    # --------------------------------------------------------
    # HORÁRIO
    # --------------------------------------------------------

    texto_nome = re.sub(
        r"\b\d{1,2}(?::\d{2})?\s*(?:h|hs|horas)\b",
        " ",
        texto_nome,
    )

    # --------------------------------------------------------
    # PALAVRAS DE CONTEXTO
    # --------------------------------------------------------

    texto_nome = re.sub(
        r"\bàs\b|\bas\b|\bem\b|\bpara\b|\bno\b|\bna\b|"
        r"\bnos\b|\bnas\b|\bdo\b|\bda\b|\bdos\b|\bdas\b",
        " ",
        texto_nome,
    )

    # --------------------------------------------------------
    # ARTIGOS NO INÍCIO
    # --------------------------------------------------------

    texto_nome = re.sub(
        r"^\s*(uma|um|a|o)\s+",
        "",
        texto_nome,
    )

    # --------------------------------------------------------
    # PONTUAÇÃO
    # --------------------------------------------------------

    texto_nome = re.sub(
        r"[?!.,;:]+",
        " ",
        texto_nome,
    )

    # --------------------------------------------------------
    # ESPAÇOS
    # --------------------------------------------------------

    texto_nome = re.sub(
        r"\s+",
        " ",
        texto_nome,
    ).strip()

    # --------------------------------------------------------
    # SOBRAS DE PERGUNTAS
    # --------------------------------------------------------

    texto_nome = re.sub(
        r"^(é|será|sera|vai ser)\s+",
        "",
        texto_nome,
    ).strip()

    # --------------------------------------------------------
    # EXPRESSÕES DE CONSULTAS GENÉRICAS
    # --------------------------------------------------------

    expressoes_consulta = [
        r"\balguma\b",
        r"\balgum\b",
        r"\bpróxima\b",
        r"\bpróximo\b",
        r"\bpróximas\b",
        r"\bpróximos\b",
        r"\bquando\b",
        r"\bqual\b",
        r"\bquais\b",
        r"\bque\b",
        r"\beventos?\b",
        r"\bmeu\b",
        r"\bmeus\b",
        r"\bminha\b",
        r"\bminhas\b",
        r"\bos\b",
        r"\bas\b",
        r"\bum\b",
        r"\buma\b",
    ]

    for expressao in expressoes_consulta:
        texto_nome = re.sub(
            expressao,
            " ",
            texto_nome,
            flags=re.IGNORECASE,
        )

    texto_nome = re.sub(
        r"\s+",
        " ",
        texto_nome,
    ).strip()

    # --------------------------------------------------------
    # PERGUNTAS GENÉRICAS
    # --------------------------------------------------------

    pergunta_generica = (
        "quais eventos" in mensagem
        or "quais os meus eventos" in mensagem
        or "quais os meus próximos eventos" in mensagem
        or "quais meus próximos eventos" in mensagem
        or "meus próximos eventos" in mensagem
        or "meus proximos eventos" in mensagem
        or "próximos eventos" in mensagem
        or "proximos eventos" in mensagem
        or "listar eventos" in mensagem
        or "liste os eventos" in mensagem
        or "mostrar eventos" in mensagem
        or "mostre os eventos" in mensagem
        or "eventos existem" in mensagem
        or "eventos tenho" in mensagem
    )

    if not pergunta_generica and texto_nome:
        nome_evento = texto_nome

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    print(
        f"[extrair_dados] "
        f"Nome: {nome_evento}"
    )

    print(
        f"[extrair_dados] "
        f"Data: {data_evento}"
    )

    print(
        f"[extrair_dados] "
        f"Horário: {horario_evento}"
    )

    return {
        "nome_evento": nome_evento,
        "data_evento": data_evento,
        "horario_evento": horario_evento,
        "erro_llm": "",
    }
# ============================================================
# NÓ: VALIDAÇÃO DOS DADOS DE CRIAÇÃO
# ============================================================

def validar_dados_criacao(state: Estado):
    print("\n[validar_dados_criacao] Validando dados...")

    nome = state["nome_evento"].strip()
    data = state["data_normalizada"].strip()
    horario = state["horario_evento"].strip()

    if not nome:
        print("[validar_dados_criacao] Nome ausente.")
        return {}

    if not data:
        print("[validar_dados_criacao] Data ausente.")
        return {}

    if not horario:
        print("[validar_dados_criacao] Horário ausente.")
        return {}

    print("[validar_dados_criacao] Dados válidos.")

    return {}


# ============================================================
# ROTEAMENTO DA VALIDAÇÃO DA CRIAÇÃO
# ============================================================

def decidir_validacao_criacao(state: Estado):
    nome = state["nome_evento"].strip()
    data = state["data_normalizada"].strip()
    horario = state["horario_evento"].strip()

    if nome and data and horario:
        return "dados_validos"

    return "dados_incompletos"


# ============================================================
# NÓ: CRIAÇÃO DO EVENTO
# ============================================================
def criar_evento(state: Estado):
    print("\n[criar_evento] Criando evento...")

    conn = conectar_eventos()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO eventos
        (nome, data, horario)
        VALUES (?, ?, ?)
        """,
        (
            state["nome_evento"],
            state["data_normalizada"],
            state["horario_evento"],
        ),
    )

    evento_id = cursor.lastrowid

    conn.commit()
    conn.close()

    novo_evento: Evento = {
        "id": evento_id,
        "nome": state["nome_evento"],
        "data": state["data_normalizada"],
        "horario": state["horario_evento"],
    }

    print(
        f"[criar_evento] Evento criado: "
        f"{novo_evento}"
    )

    return {
        "eventos": [novo_evento],
    }

# ============================================================
# NÓ: RESPOSTA DA CRIAÇÃO
# ============================================================

def resposta_criacao(state: Estado):
    evento = state["eventos"][0]

    data_br = datetime.strptime(
        evento["data"],
        "%Y-%m-%d",
    ).strftime("%d/%m/%Y")

    return {
        "resposta": (
            "Evento criado com sucesso:\n"
            f"- {evento['nome']} | "
            f"{data_br} às {evento['horario']}"
        )
    }


# ============================================================
# NÓ: SOLICITAÇÃO DE DADOS AUSENTES
# ============================================================

def solicitar_dados_criacao(state: Estado):
    faltantes = []

    if not state["nome_evento"].strip():
        faltantes.append("nome do evento")

    if not state["data_normalizada"].strip():
        faltantes.append("data")

    if not state["horario_evento"].strip():
        faltantes.append("horário")

    return {
        "resposta": (
            "Não foi possível criar o evento. "
            "Faltam: "
            + ", ".join(faltantes)
            + "."
        )
    }


# ============================================================
# ROTEAMENTO APÓS ANÁLISE DA INTENÇÃO
# ============================================================

def decidir_apos_analisar_intencao(state: Estado):
    if state["erro_llm"]:
        return "erro"

    if state["intencao"] == "desconhecida":
        return "desconhecida"

    return "continuar"

# ============================================================
# ROTEAMENTO APÓS EXTRAÇÃO
# ============================================================

def decidir_apos_extracao(state: Estado):
    if state["erro_llm"]:
        return "erro"

    return "continuar"


# ============================================================
# NÓ: CONSULTA DE EVENTOS
# ============================================================
def consultar_eventos(state: Estado):
    print("\n[consultar_eventos] Buscando eventos...")

    nome_busca = state["nome_evento"].lower().strip()
    data_busca = state["data_normalizada"]

    todos_eventos = listar_eventos()

    eventos_encontrados = []

    for evento in todos_eventos:
        nome_evento = evento["nome"].lower()

        # ----------------------------------------------------
        # FILTRO POR NOME
        # ----------------------------------------------------

        if nome_busca:
            if nome_busca not in nome_evento:
                continue

        # ----------------------------------------------------
        # FILTRO POR DATA
        # ----------------------------------------------------

        if data_busca:
            if evento["data"] != data_busca:
                continue

        eventos_encontrados.append(evento)

    # --------------------------------------------------------
    # ORDENAÇÃO
    # --------------------------------------------------------

    eventos_encontrados.sort(
        key=lambda evento: (
            evento["data"],
            evento["horario"],
        )
    )

    print(
        f"[consultar_eventos] "
        f"Eventos encontrados: {len(eventos_encontrados)}"
    )

    return {
        "eventos": eventos_encontrados,
    }


# ============================================================
# NÓ DE PASSAGEM ANTES DA VALIDAÇÃO
# ============================================================

def validar_no(state: Estado):
    print(
        "\n[validar_no] "
        "Validando resultado da consulta..."
    )

    return {}


# ============================================================
# ROTEAMENTO DO RESULTADO DA CONSULTA
# ============================================================

def validar_resultado(state: Estado):
    print(
        f"[validar_resultado] "
        f"Eventos encontrados: "
        f"{len(state['eventos'])}"
    )

    if state["eventos"]:
        return "encontrado"

    if state["tentativas"] < 2:
        return "vazio"

    return "limite_atingido"


# ============================================================
# NÓ: BUSCA ALTERNATIVA
# ============================================================

def buscar_alternativa(state: Estado):
    print(
        "\n[buscar_alternativa] "
        "Buscando eventos futuros como alternativa..."
    )

    hoje = date.today().isoformat()

    todos_eventos = listar_eventos()

    eventos_futuros = [
        evento
        for evento in todos_eventos
        if evento["data"] >= hoje
    ]

    eventos_futuros.sort(
        key=lambda evento: (
            evento["data"],
            evento["horario"],
        )
    )

    print(
        f"[buscar_alternativa] "
        f"Eventos futuros encontrados: "
        f"{len(eventos_futuros)}"
    )

    return {
        "eventos": eventos_futuros,
        "busca_alternativa": True,
        "tentativas": state["tentativas"] + 1,
    }

# ============================================================
# NÓ: RESPOSTA COM EVENTOS
# ============================================================

def resposta_final(state: Estado):
    linhas = []

    for evento in state["eventos"]:
        data_br = datetime.strptime(
            evento["data"],
            "%Y-%m-%d",
        ).strftime("%d/%m/%Y")

        linhas.append(
            f"- {evento['nome']} | "
            f"{data_br} às {evento['horario']}"
        )

    if state["busca_alternativa"]:
        introducao = (
            "Não encontrei eventos para a data solicitada. "
            "Aqui estão alguns eventos futuros:"
        )
    else:
        introducao = "Eventos encontrados:"

    return {
        "resposta": (
            introducao
            + "\n"
            + "\n".join(linhas)
        )
    }


# ============================================================
# NÓ: RESPOSTA SEM RESULTADOS
# ============================================================

def resposta_sem_resultados(state: Estado):
    data_alvo = state["data_normalizada"]

    if data_alvo:
        data_br = datetime.strptime(
            data_alvo,
            "%Y-%m-%d",
        ).strftime("%d/%m/%Y")

        mensagem = (
            f"Nenhum evento foi encontrado "
            f"para {data_br}. "
        )
    else:
        mensagem = "Nenhum evento foi encontrado. "

    if state["nome_evento"].strip():
        mensagem += (
            "Também não encontrei eventos futuros "
            f"correspondentes a "
            f"'{state['nome_evento']}'."
        )
    else:
        mensagem += (
            "Não há eventos futuros disponíveis "
            "para sugerir no momento."
        )

    return {
        "resposta": mensagem,
    }


# ============================================================
# ROTEAMENTO APÓS NORMALIZAÇÃO DA DATA
# ============================================================

def decidir_apos_normalizar_data(state: Estado):
    if state["intencao"] == "criar":
        return "criar"

    if state["intencao"] == "cancelar":
        return "cancelar"

    return "consultar"


# ============================================================
# NÓ: CANCELAMENTO DO EVENTO
# ============================================================

def cancelar_evento(state: Estado):
    print("\n[cancelar_evento] Cancelando evento...")

    nome_alvo = state["nome_evento"].strip().lower()
    data_alvo = state["data_normalizada"].strip()

    todos_eventos = listar_eventos()

    encontrados = [
        evento
        for evento in todos_eventos
        if (
            nome_alvo
            and all(
                termo in evento["nome"].lower()
                for termo in nome_alvo.split()
            )
        )
        and (
            not data_alvo
            or evento["data"] == data_alvo
        )
    ]

    if not encontrados:
        print(
            "[cancelar_evento] "
            "Nenhum evento correspondente encontrado."
        )

        return {
            "eventos": [],
        }

    evento = encontrados[0]

    conn = conectar_eventos()

    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM eventos
        WHERE id = ?
        """,
        (evento["id"],),
    )

    conn.commit()
    conn.close()

    print(
        f"[cancelar_evento] "
        f"Evento removido: {evento}"
    )

    return {
        "eventos": [evento],
    }


# ============================================================
# NÓ: RESPOSTA DO CANCELAMENTO
# ============================================================

def resposta_cancelamento(state: Estado):
    if not state["eventos"]:
        return {
            "resposta": (
                "Não foi possível cancelar o evento."
            )
        }

    evento = state["eventos"][0]

    data_br = datetime.strptime(
        evento["data"],
        "%Y-%m-%d",
    ).strftime("%d/%m/%Y")

    return {
        "resposta": (
            "Evento cancelado com sucesso:\n"
            f"- {evento['nome']} | "
            f"{data_br} às {evento['horario']}"
        )
    }


# ============================================================
# NÓ: ERRO DA LLM
# ============================================================

def resposta_erro_llm(state: Estado):
    return {
        "resposta": (
            f"{state['erro_llm']}\n"
            "O restante do fluxo não foi executado."
        )
    }


# ============================================================
# NÓ: INTENÇÃO NÃO IMPLEMENTADA
# ============================================================

def intencao_nao_implementada(state: Estado):
    print(
        "\n[intencao_nao_implementada] "
        "A intenção identificada não pertence às operações "
        "implementadas pelo sistema."
    )

    return {
        "resposta": (
            f"A intenção '{state['intencao']}' "
            "não corresponde a uma operação de "
            "gerenciamento de eventos implementada."
        )
    }



# ============================================================
# CONSTRUÇÃO DO GRAFO
# ============================================================

def construir_grafo(modo="teste", checkpointer=None):
    if modo == "teste":
        fn_intencao = analisar_intencao_teste
    elif modo == "local":
        fn_intencao = analisar_intencao_ollama
    else:
        fn_intencao = analisar_intencao_llm

    fn_extrair = extrair_dados_teste

    builder = StateGraph(Estado)

    builder.add_node("intencao_nao_implementada", intencao_nao_implementada)
    builder.add_node("analisar_intencao", fn_intencao)
    builder.add_node("extrair_dados", fn_extrair)
    builder.add_node("normalizar_data", normalizar_data)
    builder.add_node("normalizar_horario", normalizar_horario)
    builder.add_node("validar_dados_criacao", validar_dados_criacao)
    builder.add_node("criar_evento", criar_evento)
    builder.add_node("resposta_criacao", resposta_criacao)
    builder.add_node("solicitar_dados_criacao", solicitar_dados_criacao)
    builder.add_node("consultar_eventos", consultar_eventos)
    builder.add_node("validar_no", validar_no)
    builder.add_node("buscar_alternativa", buscar_alternativa)
    builder.add_node("resposta_final", resposta_final)
    builder.add_node("resposta_sem_resultados", resposta_sem_resultados)
    builder.add_node("cancelar_evento", cancelar_evento)
    builder.add_node("resposta_cancelamento", resposta_cancelamento)
    builder.add_node("resposta_erro_llm", resposta_erro_llm)

    builder.add_edge(START, "analisar_intencao")

    builder.add_conditional_edges(
        "analisar_intencao",
        decidir_apos_analisar_intencao,
        {
            "continuar": "extrair_dados",
            "desconhecida": "intencao_nao_implementada",
            "erro": "resposta_erro_llm",
        },
    )

    builder.add_conditional_edges(
        "extrair_dados",
        decidir_apos_extracao,
        {
            "continuar": "normalizar_data",
            "erro": "resposta_erro_llm",
        },
    )

    builder.add_conditional_edges(
        "normalizar_data",
        decidir_apos_normalizar_data,
        {
            "consultar": "consultar_eventos",
            "criar": "normalizar_horario",
            "cancelar": "cancelar_evento",
        },
    )

    builder.add_edge("normalizar_horario", "validar_dados_criacao")

    builder.add_conditional_edges(
        "validar_dados_criacao",
        decidir_validacao_criacao,
        {
            "dados_validos": "criar_evento",
            "dados_incompletos": "solicitar_dados_criacao",
        },
    )

    builder.add_edge("criar_evento", "resposta_criacao")
    builder.add_edge("resposta_criacao", END)
    builder.add_edge("solicitar_dados_criacao", END)

    builder.add_edge("cancelar_evento", "resposta_cancelamento")
    builder.add_edge("resposta_cancelamento", END)

    builder.add_edge("consultar_eventos", "validar_no")

    builder.add_conditional_edges(
        "validar_no",
        validar_resultado,
        {
            "encontrado": "resposta_final",
            "vazio": "buscar_alternativa",
            "limite_atingido": "resposta_sem_resultados",
        },
    )

    builder.add_edge("buscar_alternativa", "validar_no")
    builder.add_edge("resposta_final", END)
    builder.add_edge("resposta_sem_resultados", END)
    builder.add_edge("resposta_erro_llm", END)

    if checkpointer is None:
        conn_ckpt = sqlite3.connect(":memory:", check_same_thread=False)
        checkpointer = SqliteSaver(conn_ckpt)

    return builder.compile(checkpointer=checkpointer)


# ============================================================
# EXECUÇÃO INTERATIVA
# ============================================================

if __name__ == "__main__":
    if MODO_TESTE:
        modo = "teste"
        print(
            "\n===================================================="
        )
        print(" MODO DE TESTE ATIVADO")
        print(" Nenhuma LLM será utilizada.")
        print(
            "====================================================\n"
        )

    elif MODO_LOCAL:
        modo = "local"
        print(
            "\n===================================================="
        )
        print(" MODO LOCAL ATIVADO")
        print(f" Modelo: {OLLAMA_MODELO}")
        print(" Provedor: Ollama")
        print(
            "====================================================\n"
        )

    else:
        modo = "hf"
        print(
            "\n===================================================="
        )
        print(" MODO HUGGING FACE ATIVADO")
        print(" Modelo: Qwen/Qwen3-4B-Instruct-2507")
        print(
            "====================================================\n"
        )

    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)

    checkpointer = SqliteSaver(conn)

    inicializar_banco_eventos()
    inserir_eventos_iniciais()

    print("\n[SQLite] Eventos armazenados:")
    for evento in listar_eventos():
        print(evento)

    graph = construir_grafo(modo=modo, checkpointer=checkpointer)

    print("\n[grafo] Estrutura do grafo:")
    print(
        graph.get_graph().draw_ascii()
    )

    while True:
        mensagem = input(
            "\nDigite sua mensagem (ou 'sair'): "
        )

        if mensagem.lower() == "sair":
            break

        config = {
            "configurable": {
                "thread_id": "usuario-db"
            }
        }

        estado_anterior = graph.get_state(config)

        print("\n[Checkpoint anterior]")
        print(f"Estado anterior: {estado_anterior.values}")

        estado_salvo = graph.get_state(config)

        print("\n[Checkpoint]")
        print(f"Thread ID: {config['configurable']['thread_id']}")
        print(f"Estado salvo: {estado_salvo.values}")

        resultado = graph.invoke(
            {
                "mensagem": mensagem,
                "intencao": "",
                "nome_evento": "",
                "data_evento": "",
                "horario_evento": "",
                "eventos": [],
                "tentativas": 0,
                "resposta": "",
                "data_normalizada": "",
                "busca_alternativa": False,
                "erro_llm": "",
            },
            config,
        )

        estado_salvo = graph.get_state(config)

        print("\n[Checkpoint]")
        print(f"Thread ID: {config['configurable']['thread_id']}")
        print(f"Estado salvo: {estado_salvo.values}")

        print("\n[Resultado final]")

        print(
            f"Mensagem: "
            f"{resultado['mensagem']}"
        )

        print(
            f"Intenção: "
            f"{resultado['intencao']}"
        )

        print(
            f"Nome do evento: "
            f"{resultado['nome_evento']}"
        )

        print(
            f"Data: "
            f"{resultado['data_evento']}"
        )

        print(
            f"Horário: "
            f"{resultado['horario_evento']}"
        )

        print(
            f"Resposta: "
            f"{resultado['resposta']}"
        )

        print(
            f"Tentativas: "
            f"{resultado['tentativas']}"
        )

        if resultado["erro_llm"]:
            print(
                f"Erro LLM: "
                f"{resultado['erro_llm']}"
            )
