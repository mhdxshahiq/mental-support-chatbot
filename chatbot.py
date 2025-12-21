import os
import joblib

import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnableSequence

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

#Load models
clf = joblib.load("models/department_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_output_tokens=200,
    google_api_key=api_key,
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory="./chroma_db_hf",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})

chat_history = []

def predict_department(text):
    x_vec = vectorizer.transform([text])
    prediction = clf.predict(x_vec)[0]
    return prediction

def save_routing_to_json(user_query, department):
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/routing_log.json"

    # Load old logs
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    # Create new entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": user_query,
        "department": department
    }

    logs.append(entry)

    # Save updated logs
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)





prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent. Use the context below."),
    ("system", "Chat history:\n{history}"),
    ("system", "Context from database:\n{context}"),
    ("human", "{question}")
])

# def get_context_with_ids(question):
#     docs = db.similarity_search(question, k=3)

#     print("\n--- Retrieved Tickets Used ---")
#     for i, d in enumerate(docs, 1):
#         print(f"\nTicket {i}:")
#         print("ID:", d.metadata.get("ticket_id", "N/A"))
#         print("Content:", d.page_content[:200], "...\n")

#     return "\n\n".join([d.page_content for d in docs])


# FIXED RAG CHAIN
qa_chain = (
    RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "history": lambda x: x["history"],
        "question": lambda x: x["question"]
    })| prompt | model
)


# CHAT LOOP
print("Customer support chatbot. How can I help you?")

while True:
    query = input("YOU: ")
    if query.lower() in ["exit", "quit", "bye"]:
        print("BOT: Goodbye")
        break
    
    #backend log
    department = predict_department(query)
    print(f"[BACKEND ROUTING] Classified as → {department}")
    save_routing_to_json(query, department)

    # docs = retriever.invoke(query)

    # print("\n--- Retrieved Tickets (Context Used) ---")
    # for i, d in enumerate(docs, 1):
    #     print(f"\nTicket {i}:")
    #     print("ID:", d.metadata.get("ticket_id", "N/A"))
    #     print("Content:", d.page_content[:300], "...")


    # RAG
    response = qa_chain.invoke({
        "question": query,
        "history": "\n".join(chat_history)
    })

    bot_reply = response.content
    print("BOT:", bot_reply)

    # Store chat history
    chat_history.append(f"User: {query}")
    chat_history.append(f"Bot: {bot_reply}")







