# import os
# from langchain_community.document_loaders import CSVLoader
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv

# load_dotenv()

# # Embedding model
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# DB_PATH = "vector_db"

# #cvs ingection
# def ingest_csv(csv_path: str):
#     loader = CSVLoader(csv_path)
#     docs = loader.load()
#     print(f"Loaded {len(docs)} rows")

#     # Split documents
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=400,
#         chunk_overlap=80
#     )

#     chunks = splitter.split_documents(docs)
#     print(f"[INFO] Split into {len(chunks)} chunks")

#     # Save to ChromaDB
#     vectordb = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory=DB_PATH
#     )

#     vectordb.persist()
#     print("Chroma DB created successfully!")

# #chromadb
# def load_db():
#     vectordb = Chroma(
#         embedding_function=embeddings,
#         persist_directory=DB_PATH
#     )
#     return vectordb

# #chunks
# def retrieve_chunks(query: str, k: int = 3) -> str:
#     vectordb = load_db()
#     results = vectordb.similarity_search(query, k=k)
#     if not results:
#         return ""
#     combined_text = "\n\n".join([r.page_content for r in results])
#     return combined_text


import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader

DB_DIR = "chroma_db"

loader = TextLoader(
    r"C:\FILES\projects\Customer_chatbot\data\Mental_Health_FAQ.csv",
    encoding="utf-8"
)

docs = loader.load()

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if not os.path.exists(DB_DIR):
    vectorstore = Chroma.from_documents(
        collection_name="mental_health_docs",
        documents=docs,
        embedding=embed_model,
        persist_directory=DB_DIR
    )
    vectorstore.persist()
else:
    vectorstore = Chroma(
        collection_name="mental_health_docs",
        embedding_function=embed_model,
        persist_directory=DB_DIR
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def retrieve_chunks(question):
    try:
        results = retriever.invoke(question)
    except:
        results = []

    text = "\n".join([doc.page_content for doc in results]).strip()

    if text:
        return text

    q_lower = question.lower()

    for doc in docs:
        if q_lower in doc.page_content.lower():
            return doc.page_content

    return ""
