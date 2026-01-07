from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import Config


def explainer_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Educational explainer agent that generates clear explanations."""
    print("\n🎓 [EXPLAINER] Generating explanation...")
    
    llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL)
    messages = state.get("messages", [])
    query = ""
    
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            query = m.content
            break

    if not query:
        return {"messages": []}
    
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
