import streamlit as st

# Page Config
st.set_page_config("Movie Expert", page_icon=":movie_camera:")
from utils import write_message
from llm import llm, check_ollama_status, get_streaming_response, truncate_history
from graph import graph
from agent import generate_response



# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the Movie Expert Chatbot! How can I help you?"},
    ]

# Check Ollama status on startup
if not check_ollama_status():
    st.warning("⚠️ Warning: Ollama not running or Llama 3.2 model not available. Some features may not work properly.")

# Modified submit handler with streaming
def handle_submit(message):
    """Submit handler with streaming response"""
    
    # First truncate chat history to avoid context window issues
    if len(st.session_state.messages) > 10:
        st.session_state.messages = truncate_history(st.session_state.messages)
    
    # Handle the response
    with st.spinner('Thinking...'):
        try:
            # Use the streaming approach for better UX
            with st.chat_message('assistant'):
                message_placeholder = st.empty()
                full_response = ""
                
                # Call the agent to generate response
                response = generate_response(message)
                
                # If streaming fails, fall back to non-streaming
                if isinstance(response, str):
                    message_placeholder.markdown(response)
                    # Save the response
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("Error generating response. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            write_message('assistant', "I'm having trouble processing your request. Please try again with a simpler query.")


# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input("Ask me about movies..."):
    # Display user message in chat message container
    write_message('user', prompt)

    # Generate a response
    handle_submit(prompt)

# Add model info in sidebar
with st.sidebar:
    st.subheader("About this chatbot")
    st.write("This movie expert chatbot uses:")
    st.write("- Llama 3.2 via Ollama")
    st.write("- Neo4j Graph Database")
    st.write("- LangChain for orchestration")
    
    # Add a section to pull the model if needed
    st.subheader("Troubleshooting")
    if st.button("Check Ollama Status"):
        if check_ollama_status():
            st.success("✅ Ollama is running with Llama 3.2 model")
        else:
            st.error("❌ Issue with Ollama or Llama 3.2 model")
            st.code("ollama pull llama3.2", language="bash")