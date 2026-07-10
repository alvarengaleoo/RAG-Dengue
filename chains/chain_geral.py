from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Modelo da Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

# Prompt da aplicação
prompt = ChatPromptTemplate.from_template(
    """
Você é um assistente especializado em dengue.

Se o usuário fizer uma saudação, responda de forma educada e informe que pode ajudar com dúvidas sobre dengue.

Se a pergunta não estiver relacionada à dengue, informe educadamente que você responde apenas perguntas sobre esse assunto.

Pergunta:
{pergunta}
"""
)

# Chain de assuntos gerais
chain_geral = (
    prompt
    | llm
    | StrOutputParser()
)


# Teste da aplicação
if __name__ == "__main__":

    pergunta = input("Digite uma mensagem: ")

    resposta = chain_geral.invoke(
        {
            "pergunta": pergunta
        }
    )

    print("\nResposta:\n")
    print(resposta)