# RAG Dengue

Projeto desenvolvido para estudo da técnica de Retrieval-Augmented Generation (RAG) utilizando LangChain, Groq, Hugging Face Embeddings e Qdrant Cloud.

O objetivo da aplicação é responder perguntas sobre dengue com base nas informações contidas em um documento PDF.

## Tecnologias utilizadas

* Python
* LangChain
* Groq
* Hugging Face Embeddings
* Qdrant Cloud
* PyPDF
* Python Dotenv

## Funcionamento

O projeto segue o fluxo abaixo:

1. Carrega o documento PDF.
2. Divide o documento em pequenos trechos (chunks).
3. Gera embeddings para cada chunk.
4. Armazena os embeddings no Qdrant Cloud.
5. Recebe a pergunta do usuário.
6. Busca os trechos mais relevantes no banco vetorial.
7. Envia o contexto recuperado para a LLM.
8. Retorna a resposta ao usuário.

Além do fluxo RAG, o projeto possui uma etapa de classificação das perguntas para identificar se o usuário deseja informações sobre dengue ou apenas iniciou uma conversa geral.

## Estrutura do projeto

```text
RAG_Dengue/

├── banco/
│   └── indexar_documentos.py
│
├── chains/
│   ├── chain_classifica.py
│   ├── chain_geral.py
│   └── chain_rag.py
│
├── documentos/
│   └── DENGUE.pdf
│
├── config.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```

## Instalação

Criar o ambiente virtual:

```bash
python -m venv .venv
```

Ativar o ambiente (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar as dependências:

```bash
pip install -r requirements.txt
```

## Configuração

Criar um arquivo `.env` na raiz do projeto contendo:

```text
GROQ_API_KEY=sua_chave

QDRANT_URL=sua_url

QDRANT_API_KEY=sua_chave
```

## Execução

Indexar os documentos:

```bash
python -m banco.indexar_documentos
```

Executar a aplicação:

```bash
python main.py
```

## Funcionalidades

* Carregamento de documentos PDF
* Divisão dos documentos em chunks
* Geração de embeddings
* Armazenamento no Qdrant Cloud
* Busca semântica
* Respostas utilizando RAG
* Classificação de perguntas
* Respostas para saudações e perguntas fora do contexto


