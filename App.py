Openai AI Assitants import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import set_global_service_context
from llama_index.core.llms import CustomLLM
from llama_index.core.settings import Settings
from langchain_groq import ChatGroq


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables.")
    st.stop()


class GroqLLM(CustomLLM):
    model_config = {"extra": "allow"}

    def __init__(self, groq_api_key: str):
        super().__init__()
        self.groq_chat = ChatGroq(api_key=groq_api_key)

    def complete(self, prompt: str):
        """Generate a response based on input prompt."""
        response = self.groq_chat.invoke(prompt)
        return response.content

    def stream_complete(self, prompt: str):
        """Streaming response (mocked by returning full response)."""
        return self.complete(prompt)

    @property
    def metadata(self):
        """Metadata required by CustomLLM."""
        return {"model_name": "GroqLLM", "temperature": 0.7}


groq_llm = GroqLLM(groq_api_key)
Settings.llm = groq_llm


st.title("Mark")
st.write("Ask me anything!")

user_input = st.text_area("Enter your question:")

if st.button("Send"):
    if user_input.strip():
        response = groq_llm.complete(user_input)
        st.write(f"**You:** {user_input}")
        st.write(f"**Chatbot:** {response}")
    else:
        st.warning("Please enter a valid question.")

