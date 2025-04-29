import streamlit as st
from llm import llm
from graph import graph
from langchain_neo4j import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate


# Llama-optimized Cypher generation template with XML tags
CYPHER_GENERATION_TEMPLATE = """
<instructions>
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema below.
</instructions>

<rules>
- Use only the provided relationship types and properties in the schema.
- Do not use any other relationship types or properties that are not provided.
- Do not return entire nodes or embedding properties.
- For movie titles that begin with "The", move "the" to the end. For example "The 39 Steps" becomes "39 Steps, The" or "the matrix" becomes "Matrix, The".
</rules>

<schema>
{schema}
</schema>

<question>
{question}
</question>

<output_format>
Cypher Query:
</output_format>
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

# Create the Cypher QA Chain with error handling
try:
    cypher_qa = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        verbose=True,
        cypher_prompt=cypher_prompt,
        allow_dangerous_requests=True,
        return_direct=False,  # Process through LLM for better formatting
        return_intermediate_steps=True  # For debugging
    )
except Exception as e:
    st.error(f"Error initializing Cypher QA Chain: {str(e)}")
    
    # Fallback function if the chain initialization fails
    def cypher_qa(query):
        return f"I'm having trouble connecting to the movie database. Please try again later. Error: {str(e)}"