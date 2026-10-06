from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# Estado compartilhado pelo grafo
class Estado(TypedDict): # estrutura de dados que circula pelo grafo.
    mensagem: str
    categoria: str
    resposta: str


# Nó 1
def processar(state: Estado):
    mensagem = state["mensagem"].lower()

    if "urgente" in mensagem:
        categoria = "urgente"
    else:
        categoria = "normal"

    return {
        "categoria": categoria
    }


# Nó 2
def finalizar_urgente(state: Estado):
    return {
        "resposta": "A mensagem foi classificada como URGENTE."
    }


# Nó 3
def finalizar_normal(state: Estado):
    return {
        "resposta": "A mensagem foi classificada como NORMAL."
    }


# Função responsável pelo roteamento
def decidir_caminho(state: Estado):
    return state["categoria"]


# Criação do grafo
builder = StateGraph(Estado)

builder.add_node("processar", processar)
builder.add_node("finalizar_urgente", finalizar_urgente)
builder.add_node("finalizar_normal", finalizar_normal)

builder.add_edge(START, "processar")

builder.add_conditional_edges(
    "processar",
    decidir_caminho,
    {
        "urgente": "finalizar_urgente",
        "normal": "finalizar_normal",
    },
)

builder.add_edge("finalizar_urgente", END)
builder.add_edge("finalizar_normal", END)


# Compila o grafo
graph = builder.compile()


# Executa o grafo
resultado = graph.invoke({
    "mensagem": "Gostaria de informações sobre o evento.",
    "categoria": "",
    "resposta": "",
})

print(resultado)