from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import Config
from state.state_definitions import DataTeamState  # We can reuse DataTeamState or create a new one. 
# Ideally we should have a shared state or OutputTeamState. 
# For now, let's assume we might receive a generic state or specific OutputTeamState.
# Let's verify state/state_definitions.py content first? I recall seeing it in early steps.
# It had KnowledgeTeamState, RAGState, DataTeamState. 
# I should probably check if I need to add OutputTeamState. 
# For now, I'll use a generic dict or define it locally if needed, but let's check state definitions in next turn if needed.
# Actually, I'll implement it to accept a generic state dict for now, or assume it's part of the main state.

def explainer_agent_node(state: dict):
    """Educational Explainer Agent"""
    print("\n🎓 [EXPLAINER] Generating explanation...")
    
    llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL)
    
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}
    
    query = messages[-1].content
    
    # Check if we have data from other teams to enrich the explanation
    data_context = ""
    if state.get("astronomical_data"):
        data_context += f"\nData Context: {state['astronomical_data']}"
    
    system_prompt = (
        "You are an expert astronomy educator. Your goal is to explain complex "
        "astronomical concepts in simple, engaging terms suitable for students and enthusiasts. "
        "Use analogies where helpful. If data is provided, incorporate it into your explanation."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query} {data_context}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"query": query, "data_context": data_context})
    
    return {
        "messages": [AIMessage(content=response.content, name="explainer")]
    }
