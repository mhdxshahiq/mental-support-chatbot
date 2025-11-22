# 🧠 Mental Health RAG Chatbot (Gemini + LangChain + Chroma)

A lightweight mental-health conversational AI assistant built using:

- **Google Gemini 2.0 Flash**
- **LangChain**
- **ChromaDB**
- **Retrieval-Augmented Generation (RAG)**
- **Conversation Memory (last 4 turns = 8 messages)**
- **Safety filters (anti-jailbreak + topic restriction)**

This chatbot ONLY talks about emotional well-being and blocks unsafe or unrelated topics.

---

## ✨ Features

### 🔹 1. Retrieval-Augmented Generation (RAG)
Fetches relevant answers from your mental-health dataset stored in **ChromaDB**.

### 🔹 2. Conversation Memory  
Remembers the **last 4 conversation turns** (8 messages total), enabling more natural and contextual replies.

### 🔹 3. Safety Guardrails  
Prevents harmful or manipulative prompts like:

- `ignore previous`
- `jailbreak`
- `switch role`
- `system override`
- `bypass`

Also blocks off-topic questions politely.

### 🔹 4. Text Summarization  
RAG chunks are summarized before generating the final answer for better clarity.

---

## 📁 Project Structure

```
│── main.py             → Chatbot logic + safety + memory + RAG
│── rag_pipeline.py     → ChromaDB retriever
│── ingest.py           → CSV → chunks → embeddings → Chroma
│── system_prompt.py    → Base system instructions
│── data.csv            → Your mental-health FAQ dataset
│── README.md
```

---

## 🧠 Workflow (How It Works)

```
1. User enters a question
       ↓
2. Bot checks: Is it related to mental health?
       ↓
3. Retrieves relevant chunks from ChromaDB
       ↓
4. Summarizes retrieved chunks using Gemini
       ↓
5. Final prompt = Summary + Memory + User Query
       ↓
6. Gemini generates a safe, supportive response
       ↓
7. Conversation memory updates (max 4 turns)
```
