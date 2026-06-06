import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
RAG_API_KEY = os.getenv("VALID_API_KEY")

#FastAPI backend URL:
API_URL = "https://mini-rag-project-mmiz.onrender.com/query"
# API_URL = "http://localhost:8000/query"
# Page configuration:
st.set_page_config(page_title="RAG App",
   page_icon=":robot_face:",
   layout="centered"
)

# Title and description:
st.title("RAG Chatbot")
st.write("Ask questions the RAG chatbot will answer based on the documents it has been trained on.")

# API key input (optional):

user_api_key = st.sidebar.text_input("Enter API Key", type="password")
if not user_api_key:
    st.sidebar.info("No API key entered. You can still use the default free access for up to 5 queries per day.")

# Session state for chat history:

if "message" not in st.session_state:
    st.session_state["message"] = []

#Display chat history:

for msg in st.session_state["message"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#user input:

user_input = st.chat_input("Type your question here...")
if user_input:

# Add user message:

    st.session_state["message"].append(
        {"role": "user", "content": user_input}
    )
    
#Display user message:

    with st.chat_message("user"):
        st.markdown(user_input)
    
    #Assistant response:

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):

        #Send query to FastAPI backend:

            try:
                headers = {"Authorization": f"Bearer {user_api_key}"} if user_api_key else {}
                response = requests.post(
                    API_URL,
                    headers=headers,
                    json={"query": user_input}
                )
                response_data = response.json()
                if response.status_code == 200:
                    answer = response_data.get("answer", "No answer found.")
                else:
                    answer = response_data.get(
                        "detail",
                        f"API Error ({response.status_code})"
                    )
            except Exception as e:
                answer = f"Error connecting to API: {e}"
            
            st.markdown(answer)
        
    #Save assistant response to chat history:
    st.session_state["message"].append(
        {"role": "assistant", "content": answer}
    )