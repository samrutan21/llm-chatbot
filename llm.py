import streamlit as st
import requests
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

# Create LLM instance using Ollama with optimized parameters
llm = Ollama(
    model="llama3.2:latest",
    base_url="http://localhost:11434",
    temperature=0.7,
    top_p=0.9,
    num_ctx=4096,
    repeat_penalty=1.1
)

# Create the Embedding model
embeddings = OllamaEmbeddings(
    model="llama3.2:latest",
    base_url="http://localhost:11434"
)

# Retry decorator for handling connection issues
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def get_llm_response(prompt):
    """Get response from Ollama with retry logic"""
    try:
        return llm.invoke(prompt)
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to Ollama. Make sure Ollama is running on your machine.")
        return "I'm having trouble connecting to my language model. Please check if Ollama is running."
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "I encountered an error. Please try again with a simpler query."

# Cached embedding function
@st.cache_data(ttl=3600)
def cached_embedding(text):
    """Cache embeddings to improve performance"""
    return embeddings.embed_query(text)

def check_ollama_status():
    """Check if Ollama is running and has the required model"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            
            # More flexible check - look for any model that starts with "llama3.2"
            for model in models_data:
                if "llama3.2" in model["name"]:
                    # Don't show any success message in the main chat window
                    # Only return True to indicate success
                    return True
            
            # Only show warning if model is not found
            st.warning("Llama 3.2 model not found in Ollama. Run 'ollama pull llama3.2' to download.")
            return False
        return False
    except Exception as e:
        # Only show error if there's a connection issue
        st.error(f"Cannot connect to Ollama. Make sure it's running on your machine.")
        return False

def truncate_history(chat_history, max_tokens=3000):
    """Truncate chat history to avoid exceeding context window"""
    tokens = 0
    truncated_history = []
    for message in reversed(chat_history):
        # Rough estimate of tokens (1.3 tokens per word on average)
        message_tokens = len(message['content'].split()) * 1.3
        if tokens + message_tokens > max_tokens:
            break
        truncated_history.insert(0, message)
        tokens += message_tokens
    return truncated_history

# Streaming response handler
def get_streaming_response(prompt):
    """Get streaming response for Streamlit UI"""
    try:
        streaming_llm = Ollama(
            model="llama3.2",
            base_url="http://localhost:11434",
            streaming=True
        )
        return streaming_llm.stream(prompt)
    except Exception as e:
        st.error(f"Streaming error: {str(e)}")
        return ["I encountered an error while streaming. Please try again."]