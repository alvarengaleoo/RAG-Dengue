from operator import itemgetter

from langchain_core.runnables import RunnableLambda, RunnableParallel

from chains.chain_classifica import chain_classificador
from chains.chain_geral import chain_geral
from chains.chain_rag import responder


def escolher_chain(entrada):
    """
    Decide qual chain será executada de acordo com a classificação.
    """

    opcao = entrada["classificacao"].opcao

    if opcao == 1:
        print("\n>> Rota escolhida: RAG\n")
        return responder(entrada["pergunta"])

    elif opcao == 2:
        print("\n>> Rota escolhida: Assuntos Gerais\n")

        return chain_geral.invoke(
            {
                "pergunta": entrada["pergunta"]
            }
        )

    elif opcao == 3:
        return (
            "O fluxo de cadastro será implementado na próxima versão do projeto."
        )

    return "Não foi possível identificar a solicitação."


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

    print("=" * 60)
    print(" Chatbot RAG - Dengue")
    print("=" * 60)
    print("Digite 'sair' para encerrar.\n")

    while True:

        pergunta = input("Você: ")

        if pergunta.lower() == "sair":
            print("\nEncerrando o chatbot...")
            break

        resposta = router.invoke(
            {
                "pergunta": pergunta
            }
        )

        print(f"\nAssistente: {resposta}\n")


if __name__ == "__main__":
    main()