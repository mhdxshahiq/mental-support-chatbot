import streamlit as st
from main import (
    chain, get_memory_text, retrieve_chunks,
    safe_text, add_turn_to_memory, is_allowed_query
)

st.set_page_config(page_title="Mental Health Chatbot", layout="centered")

st.title("Mental Health Support Chatbot")
st.write("I'm here to talk about emotional well-being. You can ask anything related to feelings, stress, anxiety, or mental health.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.chat_history.append(("You", user_input))

    if user_input.lower() in ["exit", "quit"]:
        bot_reply = "Thank you for spending time with me. Take care."
        st.session_state.chat_history.append(("Bot", bot_reply))
        add_turn_to_memory(user_input, bot_reply)

    elif user_input.lower() in ["hi", "hello", "hey", "ok", "okay", "hmm"]:
        bot_reply = "I'm here. How are you feeling?"
        st.session_state.chat_history.append(("Bot", bot_reply))
        add_turn_to_memory(user_input, bot_reply)

    elif not is_allowed_query(user_input):
        bot_reply = "I can talk only about emotional and mental well-being."
        st.session_state.chat_history.append(("Bot", bot_reply))
        add_turn_to_memory(user_input, bot_reply)

    else:
        if user_input.lower() in ["continue", "explain", "more", "tell me more"]:
            user_input = "continue from earlier conversation."

        context = retrieve_chunks(user_input)
        context = safe_text(context)

        response = chain.invoke({
            "memory": get_memory_text(),
            "text": context,
            "question": user_input
        })

        st.session_state.chat_history.append(("Bot", response))
        add_turn_to_memory(user_input, response)

for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)
