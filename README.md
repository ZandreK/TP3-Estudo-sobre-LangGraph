# Agente de Gerenciamento de Eventos com LangGraph

Trabalho acadêmico da disciplina **Tópicos em Programação 3 (TP3)** do curso de
Ciência da Computação da UTFPR — Ponta Grossa.

## Sobre o projeto

Agente conversacional que interpreta mensagens em português e executa operações
sobre uma agenda de eventos: **criar**, **consultar** e **cancelar**. O objetivo
pedagógico é demonstrar o uso do **LangGraph** para orquestrar o fluxo entre
análise de intenção, extração de dados, validação, acesso a banco e respostas,
incluindo um ciclo de busca alternativa quando a primeira consulta não retorna
resultados.

## Tecnologias

- Python
- LangGraph (grafo de estados e checkpointer)
- SQLite (persistência dos eventos e dos checkpoints)
- Ollama + Qwen3 1.7B (LLM local)
- Hugging Face Inference API (LLM alternativo)

## Arquitetura

A separação de responsabilidades segue:

- **LLM** — interpreta a linguagem natural e classifica a intenção.
- **Python** — extrai, normaliza e valida dados estruturados (datas, horários, nomes).
- **LangGraph** — controla o fluxo, o estado e o ciclo de busca alternativa.
- **SQLite** (`eventos.db`) — fonte de verdade dos eventos.
- **Checkpointer** (`checkpoints.db`) — persistência do estado da execução do grafo.

Ambos os bancos são criados automaticamente na primeira execução.

## Pré-requisitos

- Python 3.10 ou superior
- (Opcional) [Ollama](https://ollama.com/) instalado e rodando, para o MODO_LOCAL
- (Opcional) Token da Hugging Face, para o MODO_HF

## Instalação

```bash
git clone <url-do-repositório>
cd langGraph

python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## Configuração

Copie o arquivo de exemplo e preencha o token (apenas se for usar o MODO_HF):

```bash
cp .env.example .env
```

Edite `.env`:

```
token_teste=hf_seu_token_aqui
```

> O arquivo `.env` está no `.gitignore` e não deve ser publicado.

### Ollama (MODO_LOCAL)

```bash
# Baixar o modelo
ollama pull qwen3:1.7b

# Verificar que o serviço está no ar
curl http://localhost:11434/api/tags
```

O código espera o endpoint padrão `http://localhost:11434/api/chat`.

## Execução

O modo de execução é controlado pelas constantes no topo de `ex1.py`:

```python
MODO_TESTE = False     # True = determinístico, sem LLM
MODO_LOCAL = True      # True = usa Ollama; False + MODO_TESTE=False usa Hugging Face
```

Em seguida:

```bash
python ex1.py
```

O programa inicia um loop interativo. Para sair, digite `sair`.

### Exemplos de mensagens

- `Quais eventos tenho amanhã?`
- `Quero cadastrar uma reunião amanhã às 11h`
- `Cancele a reunião de teste amanhã`
- `Quando é a palestra de inteligência artificial?`

## Testes

A suíte automatizada executa 6 cenários nos três modos:

```bash
python testes.py
```

Saída esperada (quando todos os provedores estão disponíveis):

```
MODO_TESTE: 6/6
MODO_LOCAL: 6/6
MODO_HF: 6/6
```

Cenários cobertos:

1. Consulta de eventos
2. Criação de evento
3. Cancelamento de evento
4. Intenção desconhecida
5. Dados incompletos (fluxo de solicitação)
6. Consulta sem resultado (exercita o ciclo do grafo)

Os testes usam **bancos SQLite temporários** e **checkpointer in-memory**, isolados
do `eventos.db` e `checkpoints.db` reais. Se Ollama ou Hugging Face não estiverem
disponíveis, o modo correspondente é marcado como `SKIP`.

## Estrutura do projeto

```
ex1.py             # Aplicação principal: grafo, nós, loop interativo e construir_grafo()
testes.py          # Suíte automatizada de testes (6 cenários x 3 modos)
requirements.txt   # Dependências
.env.example       # Modelo do arquivo de credenciais
main.py            # Exemplo mínimo de grafo (material do início da disciplina)
teste_api.py       # Teste isolado da API Hugging Face
```

## Observações

- O MODO_HF depende de quota/disponibilidade da API; o projeto já teve
  `402 Payment Required` por consumo de créditos.
- Os bancos `eventos.db` e `checkpoints.db` são criados automaticamente e
  ignorados pelo Git.
