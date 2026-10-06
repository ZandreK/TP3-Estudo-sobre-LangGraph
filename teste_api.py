
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

token = os.getenv("token_teste")

if not token:
    raise RuntimeError("token_teste não encontrado no arquivo .env")

client = InferenceClient(
    api_key=token,
)

resposta = client.chat.completions.create(
    model="Qwen/Qwen3-4B-Instruct-2507",
    messages=[
        {
            "role": "system",
            "content": """
            Você é um classificador de intenções de um sistema de gerenciamento de eventos.

            Sua única tarefa é identificar a intenção da mensagem do usuário.

            As intenções possíveis são:
            - consultar
            - criar
            - cancelar
            - desconhecida

            Regras:
              - "consultar": buscar ou listar eventos existentes.
              - "criar": cadastrar ou adicionar um evento.
              - "cancelar": cancelar ou remover um evento.
              - "desconhecida": quando a intenção não se encaixar nas anteriores.

            Responda exclusivamente com uma das quatro palavras:
            consultar, criar, cancelar ou desconhecida.

            Não faça perguntas.
            Não explique sua resposta.
            """
        },
        {
            "role": "user",
            "content": "Quero consultar meus eventos de amanhã.",
        },
    ],
    max_tokens=10,
)

print(resposta.choices[0].message.content)
