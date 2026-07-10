from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnableParallel
from chains.chain_classifica import chain_classificador
from chains.chain_geral import chain_geral
from chains.chain_rag import responder


# Escolhe qual chain será executada
def escolher_chain(entrada):

    opcao = entrada["classificacao"].opcao

    if opcao == 1:
        return responder(entrada["pergunta"])

    elif opcao == 2:
        return chain_geral.invoke(
            {
                "pergunta": entrada["pergunta"]
            }
        )

    elif opcao == 3:
        return "O cadastro de ocorrência será implementado em uma próxima versão."

    return "Não foi possível processar a solicitação."


# Router principal
router = (
    RunnableParallel(
        {
            "pergunta": itemgetter("pergunta"),
            "classificacao": itemgetter("pergunta") | chain_classificador,
        }
    )
    | RunnableLambda(escolher_chain)
)


def main():

    print("=" * 50)
    print("Chatbot RAG - Dengue")
    print("=" * 50)
    print("Digite 'sair' para encerrar.\n")

    while True:

        pergunta = input("Você: ")

        if pergunta.lower() == "sair":
            print("\nAté a próxima!")
            break

        resposta = router.invoke(
            {
                "pergunta": pergunta
            }
        )

        print(f"\nAssistente: {resposta}\n")


# Inicia a aplicação
if __name__ == "__main__":
    main()