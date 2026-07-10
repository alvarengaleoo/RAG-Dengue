from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()


class ClassificaEntrada(BaseModel):

    opcao: int = Field(
        description="""
1 = Pergunta sobre dengue.

2 = Saudação ou conversa geral.

3 = Cadastro de ocorrência.
"""
    )


parser = PydanticOutputParser(
    pydantic_object=ClassificaEntrada
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Você é um classificador de perguntas.

Escolha apenas uma opção.

1 - Perguntas sobre dengue.

2 - Saudações ou assuntos gerais.

3 - Cadastro de ocorrência de dengue.

{format_instructions}
"""
        ),
        (
            "human",
            "{pergunta}"
        )
    ]
)

chain_classificador = (
    prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    | llm
    | parser
)

if __name__ == "__main__":

    pergunta = input("Digite uma pergunta: ")

    resposta = chain_classificador.invoke(
        {
            "pergunta": pergunta
        }
    )

    print()
    print(resposta)
    print()
    print("Opção:", resposta.opcao)