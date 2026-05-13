import streamlit as st
from rag_chain import load_chain

# Page config
st.set_page_config(
    page_title="Medical Q&A Assistant",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Medical Q&A Assistant")
st.markdown("Ask any medical question. Answers are grounded in real medical documents.")
st.divider()

# Load the chain once and cache it so it doesn't reload every time
@st.cache_resource
def get_chain():
    return load_chain()

chain = get_chain()

# Chat history — stores conversation in the session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input box at the bottom
if question := st.chat_input("Ask a medical question..."):

    # Show user's question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Get answer from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Searching medical documents..."):
            answer = chain.run(question)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})