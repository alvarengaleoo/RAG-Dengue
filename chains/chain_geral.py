from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Carrega as variáveis do .env
load_dotenv()

# Instancia a LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

# Prompt da Chain
prompt = ChatPromptTemplate.from_template(
    """
Você é um assistente especializado em dengue.

Sua função é responder apenas assuntos relacionados à dengue.

Regras:

- Se o usuário apenas cumprimentar, responda de forma educada e amigável.
- Explique que você pode responder dúvidas sobre sintomas, transmissão, prevenção e tratamento da dengue.
- Se o usuário perguntar sobre assuntos que não tenham relação com dengue, informe educadamente que você é especializado apenas nesse tema.
- Nunca invente informações.

Pergunta do usuário:

{pergunta}
"""
)

# Criação da Chain
chain_geral = (
    prompt
    | llm
    | StrOutputParser()
)

# Teste
if __name__ == "__main__":

    pergunta = input("Digite uma mensagem: ")

    resposta = chain_geral.invoke(
        {
            "pergunta": pergunta
        }
    )

    print("\n===== RESPOSTA =====\n")
    print(resposta)