import io
import os
import sqlite3
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import date, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import ex1
from ex1 import (
    construir_grafo,
    conectar_eventos,
    inicializar_banco_eventos,
    listar_eventos,
)


# ============================================================
# UTILITÁRIOS
# ============================================================

def estado_inicial(mensagem):
    return {
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
    }


def executar_grafo(graph, mensagem, thread_id):
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }
    return graph.invoke(estado_inicial(mensagem), config)


def inserir_evento_teste(nome, data, horario):
    conn = conectar_eventos()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO eventos (nome, data, horario) VALUES (?, ?, ?)",
        (nome, data, horario),
    )
    conn.commit()
    conn.close()


def contar_eventos(nome=None):
    conn = conectar_eventos()
    cursor = conn.cursor()
    if nome:
        cursor.execute(
            "SELECT COUNT(*) FROM eventos WHERE nome LIKE ?",
            (f"%{nome}%",),
        )
    else:
        cursor.execute("SELECT COUNT(*) FROM eventos")
    total = cursor.fetchone()[0]
    conn.close()
    return total


def verificar_ollama():
    try:
        req = Request(
            "http://localhost:11434/api/tags",
            method="GET",
        )
        with urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def verificar_hf():
    if ex1.client is None:
        return False
    try:
        ex1.client.chat.completions.create(
            model="Qwen/Qwen3-4B-Instruct-2507",
            messages=[
                {"role": "user", "content": "teste"},
            ],
            max_tokens=1,
        )
        return True
    except Exception:
        return False


# ============================================================
# BANCO TEMPORÁRIO
# ============================================================

def criar_banco_temporario():
    fd, caminho = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    ex1.BANCO_EVENTOS = caminho
    inicializar_banco_eventos()
    return caminho


def remover_banco_temporario(caminho):
    ex1.BANCO_EVENTOS = "eventos.db"
    try:
        os.unlink(caminho)
    except OSError:
        pass


# ============================================================
# TESTES
# ============================================================

def teste_consulta(graph):
    amanha = (date.today() + timedelta(days=1)).isoformat()
    inserir_evento_teste("Palestra de teste", amanha, "19:00")

    resultado = executar_grafo(
        graph,
        "Quais eventos tenho amanhã?",
        "teste-consulta",
    )

    assert resultado["intencao"] == "consultar", (
        f"Intenção esperada 'consultar', obtida '{resultado['intencao']}'"
    )

    assert len(resultado["eventos"]) > 0, (
        "Deveria encontrar pelo menos um evento para amanhã"
    )

    assert resultado["resposta"], (
        "Resposta não deve estar vazia"
    )


def teste_criacao(graph):
    contagem_antes = contar_eventos()

    resultado = executar_grafo(
        graph,
        "Quero cadastrar uma reunião amanhã às 11h",
        "teste-criacao",
    )

    assert resultado["intencao"] == "criar", (
        f"Intenção esperada 'criar', obtida '{resultado['intencao']}'"
    )

    assert resultado["nome_evento"], (
        "Nome do evento não foi extraído"
    )

    assert resultado["data_normalizada"], (
        "Data não foi normalizada"
    )

    assert resultado["horario_evento"], (
        "Horário não foi normalizado"
    )

    contagem_depois = contar_eventos()
    assert contagem_depois == contagem_antes + 1, (
        f"Evento não foi inserido no SQLite "
        f"(antes: {contagem_antes}, depois: {contagem_depois})"
    )

    assert "criado" in resultado["resposta"].lower(), (
        "Resposta deveria indicar sucesso na criação"
    )


def teste_cancelamento(graph):
    amanha = (date.today() + timedelta(days=1)).isoformat()
    inserir_evento_teste("reunião de teste", amanha, "15:00")

    assert contar_eventos("reunião de teste") > 0, (
        "Evento de teste deveria existir antes do cancelamento"
    )

    resultado = executar_grafo(
        graph,
        "Quero cancelar a reunião de teste amanhã",
        "teste-cancelamento",
    )

    assert resultado["intencao"] == "cancelar", (
        f"Intenção esperada 'cancelar', obtida '{resultado['intencao']}'"
    )

    assert contar_eventos("reunião de teste") == 0, (
        "Evento deveria ter sido removido do SQLite"
    )

    assert resultado["resposta"], (
        "Resposta não deve estar vazia"
    )


def teste_intencao_desconhecida(graph):
    contagem_antes = contar_eventos()

    resultado = executar_grafo(
        graph,
        "Me explique como funciona Python",
        "teste-desconhecida",
    )

    assert resultado["intencao"] == "desconhecida", (
        f"Intenção esperada 'desconhecida', obtida '{resultado['intencao']}'"
    )

    contagem_depois = contar_eventos()
    assert contagem_depois == contagem_antes, (
        "Não deveria ter criado ou removido eventos"
    )

    assert resultado["resposta"], (
        "Resposta não deve estar vazia"
    )


def teste_dados_incompletos(graph):
    contagem_antes = contar_eventos()

    resultado = executar_grafo(
        graph,
        "Quero cadastrar uma reunião amanhã",
        "teste-incompleto",
    )

    assert resultado["intencao"] == "criar", (
        f"Intenção esperada 'criar', obtida '{resultado['intencao']}'"
    )

    contagem_depois = contar_eventos()
    assert contagem_depois == contagem_antes, (
        "Evento não deveria ter sido inserido (dados incompletos)"
    )

    assert "faltam" in resultado["resposta"].lower(), (
        "Resposta deveria indicar dados faltantes"
    )


def teste_consulta_sem_resultado(graph):
    resultado = executar_grafo(
        graph,
        "Quando é o evento de física quântica?",
        "teste-ciclo",
    )

    assert resultado["intencao"] == "consultar", (
        f"Intenção esperada 'consultar', obtida '{resultado['intencao']}'"
    )

    assert resultado["tentativas"] >= 1, (
        "Deveria ter tentado busca alternativa pelo menos uma vez"
    )

    assert (
        "nenhum" in resultado["resposta"].lower()
        or "não" in resultado["resposta"].lower()
    ), "Resposta deveria indicar ausência de resultados"


# ============================================================
# EXECUÇÃO DA SUÍTE
# ============================================================

TESTES = [
    ("Consulta de eventos", teste_consulta),
    ("Criação de evento", teste_criacao),
    ("Cancelamento de evento", teste_cancelamento),
    ("Intenção desconhecida", teste_intencao_desconhecida),
    ("Dados incompletos", teste_dados_incompletos),
    ("Consulta sem resultado", teste_consulta_sem_resultado),
]


def executar_modo(modo):
    if modo == "local" and not verificar_ollama():
        return None, "Ollama indisponível"

    if modo == "hf" and not verificar_hf():
        return None, "API Hugging Face indisponível"

    resultados = []

    for nome_teste, funcao_teste in TESTES:
        banco_temp = criar_banco_temporario()
        try:
            graph = construir_grafo(modo=modo)
            with redirect_stdout(io.StringIO()):
                funcao_teste(graph)
            resultados.append((nome_teste, True, None))
            print(f"  [OK] {nome_teste}")
        except AssertionError as e:
            resultados.append((nome_teste, False, str(e)))
            print(f"  [FALHOU] {nome_teste}")
            print(f"           {e}")
        except Exception as e:
            resultados.append((nome_teste, False, str(e)))
            print(f"  [ERRO] {nome_teste}")
            print(f"         {type(e).__name__}: {e}")
        finally:
            remover_banco_temporario(banco_temp)

    return resultados, None


def main():
    print("=" * 50)
    print("TESTES AUTOMATIZADOS - LANGGRAPH")
    print("=" * 50)

    modos = ["teste", "local", "hf"]
    resumo = {}

    for modo in modos:
        nome_modo = f"MODO_{modo.upper()}"
        print(f"\n{nome_modo}\n")

        resultados, motivo_skip = executar_modo(modo)

        if motivo_skip:
            print(f"  [SKIP] {motivo_skip}")
            resumo[nome_modo] = "SKIP"
        else:
            passou = sum(1 for _, ok, _ in resultados if ok)
            total = len(resultados)
            resumo[nome_modo] = f"{passou}/{total}"
            print(f"\n  Resultado: {passou}/{total} testes passaram")

    print("\n" + "=" * 50)
    print("RESUMO")
    print("=" * 50)

    for modo, resultado in resumo.items():
        print(f"  {modo}: {resultado}")

    print()

    todos_ok = all(
        "SKIP" in r or r.split("/")[0] == r.split("/")[1]
        for r in resumo.values()
    )

    sys.exit(0 if todos_ok else 1)


if __name__ == "__main__":
    main()
