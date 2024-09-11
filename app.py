import streamlit as st

from rag import process_user_input

st.set_page_config(
    page_title="RAG Bot",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("RAG Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # make inference using RAG
    resp = process_user_input(prompt)

    response = f"{resp['output']}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


# if __name__ == "__main__":
#     cli.main_run(["app.py", "--server.port", "8000"])
# from langchain.vectorstores import FAISS