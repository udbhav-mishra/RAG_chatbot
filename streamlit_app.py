import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
RAG_API_KEY = os.getenv("VALID_API_KEY")

#FastAPI backend URL:
API_URL = "https://mini-rag-project-mmiz.onrender.com/query"

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
    st.warning("Please enter an API key to continuse.")
    st.stop()

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
                response = requests.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {user_api_key}"},
                    json={"query": user_input}
                )
                response_data = response.json()
                
                if response.status_code == 401:
                    answer = "Authentication failed. Invalid API key."
                elif response.status_code == 403:
                    answer = "Access denied."
                elif response.status_code != 200:
                    answer = f"API Error ({response.status_code})"
                else:
                    answer = response_data.get("answer", "No answer found.")
                
                if "error" in response_data:
                    answer = response_data["error"]
                else:
                    answer = response_data.get("answer", "No answer found.")
            except Exception as e:
                answer = f"Error connecting to API: {e}"
            
            st.markdown(answer)
        
    #Save assistant response to chat history:
    st.session_state["message"].append(
        {"role": "assistant", "content": answer}
    )