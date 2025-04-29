from llm import llm, get_llm_response
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from utils import get_session_id
from langchain_core.prompts import PromptTemplate
from tools.vector import get_movie_plot
from tools.cypher import cypher_qa
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential


# Llama-optimized chat prompt with XML tags
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "<role>You are a movie expert providing information about movies. Respond only with factual information based on the data provided.</role>"),
        ("human", "{input}"),
    ]
)

# Basic movie chat function
movie_chat = chat_prompt | llm | StrOutputParser()

# Define tools
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general movie chat not covered by other tools",
        func=movie_chat.invoke,
    ), 
    Tool.from_function(
        name="Movie Plot Search",  
        description="For when you need to find information about movies based on a plot",
        func=get_movie_plot, 
    ),
    Tool.from_function(
        name="Movie information",
        description="Provide information about movies questions using Cypher",
        func=cypher_qa
    )
]

# Neo4j chat history
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# Llama-optimized agent prompt with XML tags
agent_prompt = PromptTemplate.from_template("""
<instructions>
You are a movie expert providing information about movies.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to movies, actors or directors.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.
</instructions>

<tools>
You have access to the following tools:

{tools}
</tools>

<tool_usage>
To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]
</tool_usage>

<chat_history>
Previous conversation history:
{chat_history}
</chat_history>

<user_query>
New input: {input}
</user_query>

{agent_scratchpad}
""")

# Create the agent with retry logic
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def create_agent_with_retry():
    try:
        agent = create_react_agent(llm, tools, agent_prompt)
        return agent
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        raise

# Initialize agent with error handling
try:
    agent = create_agent_with_retry()
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,  # Handle parsing errors gracefully
        max_iterations=5  # Limit iterations to avoid excessive token usage
    )

    chat_agent = RunnableWithMessageHistory(
        agent_executor,
        get_memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
except Exception as e:
    st.error(f"Failed to initialize agent: {str(e)}")
    # Fallback to simple LLM response without agent capabilities
    def generate_response(user_input):
        return f"I'm currently operating in fallback mode and can only provide simple responses. Technical issue: {str(e)}"
else:
    # Define the response generation function with error handling
    def generate_response(user_input):
        """
        Create a handler that calls the Conversational agent
        and returns a response to be rendered in the UI
        """
        try:
            response = chat_agent.invoke(
                {"input": user_input},
                {"configurable": {"session_id": get_session_id()}},
            )
            return response['output']
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            # Fallback to direct LLM call
            prompt = f"As a movie expert, please answer this question concisely: {user_input}"
            return get_llm_response(prompt)