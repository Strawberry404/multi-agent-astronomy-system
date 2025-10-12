from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI


def create_agent(
    llm: ChatGoogleGenerativeAI,
    tools: list,
    system_prompt: str
) -> AgentExecutor:
    """Create a function-calling agent"""
    system_prompt += (
        "\nWork autonomously according to your specialty, using the tools available to you. "
        "Do not ask for clarification. "
        "Your other team members will collaborate with you with their own specialties."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor