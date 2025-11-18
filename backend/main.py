import os
import json
from datetime import datetime
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

#Load routing model
clf = joblib.load("models/department_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

#Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    max_output_tokens=200,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

chat_history = []


#vectorDB retriever for a department
def get_department_retriever(department):
    dep = department.lower()
    folder = os.path.join(os.path.dirname(__file__), "chroma_db", dep)

    if not os.path.exists(folder):
        raise Exception(f"Vector DB missing for department '{department}' at: {folder}")

    return Chroma(
        persist_directory=folder,
        embedding_function=embeddings
    ).as_retriever(search_kwargs={"k": 5})


# Predict department
def predict_department(text):
    x = vectorizer.transform([text])
    return clf.predict(x)[0]


# Save routing log
def save_routing_to_json(query, dept):
    os.makedirs("logs", exist_ok=True)
    file = "logs/routing_log.json"

    if os.path.exists(file):
        logs = json.load(open(file))
    else:
        logs = []

    logs.append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "department": dept
    })

    json.dump(logs, open(file, "w"), indent=4)


# RAG Chain
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are an expert customer service assistant.
      Use the provided Context and Chat History to generate a comprehensive and accurate answer to the user's Question.
      You MUST base your answer *only* on the provided Context. If the answer is genuinely not found in the Context,
      respond clearly with 'The information is not available in our knowledge base."""),
    ("system", "Chat history:\n{history}"),
    ("system", "Context:\n{context}"),
    ("human", "{question}")
])

def rag_chain(question, history, dept):
    retriever = get_department_retriever(dept)
    docs = retriever.invoke(question)

    formatted_context = "\n\n".join([d.page_content for d in docs])

    chain = prompt | llm 
    
    chain_input = {
        "question": question,
        "context": formatted_context,
        "history": history
    }

    return chain, chain_input


# FASTAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    department: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    query = request.message

    dept = predict_department(query)
    save_routing_to_json(query, dept)

    history = "\n".join(chat_history)
    
    # Get the chain and the input dictionary
    chain, chain_input = rag_chain(query, history, dept) 
    
    # Execute the chain using
    result = chain.invoke(chain_input)

    # The result will be a ChatMessage object
    reply = result.content 

    chat_history.append(f"User: {query}")
    chat_history.append(f"Bot: {reply}")

    return ChatResponse(reply=reply, department=dept)


@app.get("/")
def home():
    return {"message": "Backend running!"}
