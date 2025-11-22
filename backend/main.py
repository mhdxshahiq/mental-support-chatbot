import os
import re
from dotenv import load_dotenv
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableSequence
from system_prompt import system_prompt
from rag_pipeline import retrieve_chunks

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    max_output_tokens=250,
    google_api_key=api_key
)

DISALLOWED_PHRASES = [
    "ignore previous", "forget previous", "jailbreak",
    "system override", "switch role", "bypass"
]

conversation_memory = []

def add_turn_to_memory(user_msg, bot_msg):
    global conversation_memory
    conversation_memory.append(f"User: {user_msg}")
    conversation_memory.append(f"Bot: {bot_msg}")

    if len(conversation_memory) > 8:
        conversation_memory = conversation_memory[-8:]

def get_memory_text():
    return "\n".join(conversation_memory)

def is_allowed_query(q):
    clean = re.sub(r"[^a-zA-Z ]+", " ", q.lower())
    tokens = clean.split()

    allowed = {
        "mental", "illness", "stress", "anxiety",
        "depression", "disorder", "therapy", "counseling",
        "emotion", "feel", "panic", "hurt", "sad"
    }

    if "mental" in tokens and "illness" in tokens:
        return True

    if any(w in tokens for w in allowed):
        return True

    if conversation_memory:
        return True

    return False

class SafetyParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        lower = text.lower()
        for p in DISALLOWED_PHRASES:
            if p in lower:
                return "I can't respond to that."
        return text

parser = SafetyParser()

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize this text in 2–3 short sentences."),
    ("human", "{context}")
])

summary_chain = summary_prompt | llm | parser

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt + "\nRespond in a warm, friendly tone."),
    ("human",
     "Memory:\n{memory}\n\n"
     "Content:\n{text}\n\n"
     "User: {question}")
])

def safe_text(t):
    if not t or t.strip() == "":
        return "No relevant information found."
    return t

chain = RunnableSequence(
    lambda x: {
        "summary":
            summary_chain.invoke({"context": x["text"]})
            if x["text"].strip()
            else "No RAG info found.",
        "memory": x["memory"],
        "question": x["question"]
    },
    lambda x: {
        "memory": x["memory"],
        "question": x["question"],
        "text": safe_text(x["summary"])
    },
    prompt,
    llm,
    parser
)

print("Bot: I'm here if you want to talk. Type 'exit' anytime.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Thank you for spending time with me. Take care.")
        break

    if user_input.lower() in ["hi", "hello", "hey", "ok", "okay", "hmm"]:
        bot_reply = "I'm here. How are you feeling?"
        print("Bot:", bot_reply)
        add_turn_to_memory(user_input, bot_reply)
        continue

    if not is_allowed_query(user_input):
        bot_reply = "I can talk only about mental well-being."
        print("Bot:", bot_reply)
        add_turn_to_memory(user_input, bot_reply)
        continue

    if user_input.lower() in ["continue", "explain", "tell me more", "more"]:
        if conversation_memory:
            user_input = "continue from earlier conversation."
        else:
            user_input = "continue"

    context = retrieve_chunks(user_input)
    context = safe_text(context)

    response = chain.invoke({
        "memory": get_memory_text(),
        "text": context,
        "question": user_input
    })

    print("Bot:", response)
    add_turn_to_memory(user_input, response)
