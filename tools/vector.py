import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from graph import graph
from langchain_neo4j import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
import numpy as np

# Create LLM instance
llm = Ollama(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0.7
)

# Create the Embedding model
embeddings = OllamaEmbeddings(
    model="llama3.2",
    base_url="http://localhost:11434"
)

# Cache embeddings for better performance
@st.cache_data(ttl=3600)
def cached_embedding(text):
    """Cache embeddings to improve performance"""
    return embeddings.embed_query(text)

# Custom adapter for Ollama embeddings to Neo4j vector index
class DimensionAdapterEmbeddings:
    """Adapter class to handle dimension mismatch between embedding models"""
    
    def __init__(self, base_embeddings, target_dimension=1536):
        self.base_embeddings = base_embeddings
        self.target_dimension = target_dimension
    
    def embed_query(self, text):
        """Embed a query and adapt to target dimension"""
        # Get the original embedding
        original_embedding = self.base_embeddings.embed_query(text)
        
        # Adapt to the target dimension
        if len(original_embedding) > self.target_dimension:
            # Truncate if original is larger
            return original_embedding[:self.target_dimension]
        elif len(original_embedding) < self.target_dimension:
            # Pad with zeros if original is smaller
            return original_embedding + [0] * (self.target_dimension - len(original_embedding))
        else:
            return original_embedding
    
    def embed_documents(self, documents):
        """Embed multiple documents and adapt to target dimension"""
        original_embeddings = self.base_embeddings.embed_documents(documents)
        adapted_embeddings = []
        
        for emb in original_embeddings:
            if len(emb) > self.target_dimension:
                adapted_embeddings.append(emb[:self.target_dimension])
            elif len(emb) < self.target_dimension:
                adapted_embeddings.append(emb + [0] * (self.target_dimension - len(emb)))
            else:
                adapted_embeddings.append(emb)
                
        return adapted_embeddings

# Create the adapted embeddings
adapted_embeddings = DimensionAdapterEmbeddings(embeddings, target_dimension=1536)

# Retry decorator for neo4j vector operations
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def initialize_vector_store():
    """Initialize the vector store with error handling and dimension adaptation"""
    try:
        # Check if we can connect to Ollama
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            st.error("Cannot connect to Ollama. Make sure it's running.")
            return None
            
        # Create vector store with dimension adaption
        return Neo4jVector.from_existing_index(
            adapted_embeddings,                         # Use adapted embeddings
            graph=graph,                              
            index_name="moviePlots",                  
            node_label="Movie",                       
            text_node_property="plot",                
            embedding_node_property="plotEmbedding",  
            retrieval_query="""
        RETURN
            node.plot AS text,
            score,
            {
                title: node.title,
                directors: [ (person)-[:DIRECTED]->(node) | person.name ],
                actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
                tmdbId: node.tmdbId,
                source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
            } AS metadata
        """
        )
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        # For debugging
        st.error("Consider recreating your vector index with Ollama embeddings")
        return None

# Try to initialize the vector store
try:
    neo4jvector = initialize_vector_store()
    
    # Create retriever with fallback
    if neo4jvector:
        retriever = neo4jvector.as_retriever(
            search_kwargs={"k": 3}  # Limit to 3 results to reduce token usage
        )
        
        # Llama-optimized instructions with XML tags
        instructions = """
        <instructions>
        Use the given context to answer the question about movies.
        If you don't know the answer, say you don't know.
        Do not make up information that is not in the context.
        </instructions>

        <context>
        {context}
        </context>
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", instructions),
                ("human", "{input}"),
            ]
        )

        # Create the chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        plot_retriever = create_retrieval_chain(
            retriever, 
            question_answer_chain
        )

        # Define the function with success path
        def get_movie_plot(input):
            try:
                return plot_retriever.invoke({"input": input})
            except Exception as e:
                st.error(f"Error in vector search: {str(e)}")
                return {"answer": "I encountered an issue searching movie plots. Could you try rephrasing your question?"}
    else:
        # Fallback function if vector store initialization failed
        def get_movie_plot(input):
            return {"answer": "Vector search is currently unavailable due to an embedding dimension mismatch (Ollama: 3072, Neo4j index: 1536). Please use the Movie Information tool for movie-related questions."}
except Exception as e:
    st.error(f"Vector search initialization error: {str(e)}")
    # Fallback function if anything fails
    def get_movie_plot(input):
        return {"answer": "I'm having trouble accessing the movie plot database. Please try the Movie Information tool instead."}