from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import Config
from datetime import datetime


def observation_planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Observation Planner Agent"""
    print("\n🔭 [PLANNER] Creating observation plan...")
    
    llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL)
    
    messages = state.get("messages", [])
    query = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
             query = m.content
             break

    if not query:
        return {"messages": []}
    
    # Extract visibility info if available from Data Team
    visibility_info = state.get("visibility_info", {})
    location = state.get("location", {"lat": 40.7128, "lon": -74.0060})
    
    context = f"Observer Location: Lat {location.get('lat', 40.7)}, Lon {location.get('lon', -74.0)}\nDate: {datetime.now().strftime('%Y-%m-%d')}\n"
    if visibility_info:
        context += f"Visibility Data: {visibility_info}\n"
        
    system_prompt = (
        "You are an expert observational astronomer. Create a practical observation plan "
        "based on the user's request and available data. "
        "Include best times to view, what equipment is needed (naked eye, binoculars, telescope), "
        "and where to look in the sky."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Request: {query}\n\nContext:\n{context}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"query": query, "context": context})
    
    return {
        "messages": [AIMessage(content=response.content, name="observation_planner")]
    }
