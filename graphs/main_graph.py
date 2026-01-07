from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, BaseMessage
from typing import Annotated, List, TypedDict, Dict, Any
import operator
import functools

from agents.astronomer import astronomer_agent_node
from graphs.knowledge_team_graph import knowledge_team
from graphs.data_team_graph import data_team
from graphs.output_team_graph import output_team
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import Config

# Unified State inheriting/combining necessary fields
class MainState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    
    # Shared context fields
    # We can add specific fields from sub-states if we want to bubble them up explicitly
    # But often passing "messages" is enough for context if agents look at history.
    # However, data_team puts data in 'astronomical_data' etc.
    # We should include these so they persist.
    astronomical_data: Dict[str, Any]
    calculations: Dict[str, Any]
    visibility_info: Dict[str, Any]
    
    sources: List[str]
    citations: List[dict]
    
    # User context
    location: Dict[str, float]
    
def build_main_graph():
    llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL)
    
    workflow = StateGraph(MainState)
    
    # Add Main Astronomer Supervisor
    workflow.add_node("astronomer", functools.partial(astronomer_agent_node, llm=llm))
    
    # Add Sub-Graphs as Nodes
    # We can treat the compiled graphs as nodes
    workflow.add_node("knowledge_team", knowledge_team)
    workflow.add_node("data_team", data_team)
    workflow.add_node("output_team", output_team)
    
    # Edges
    workflow.add_conditional_edges(
        "astronomer",
        lambda state: state["next"],
        {
            "knowledge_team": "knowledge_team",
            "data_team": "data_team",
            "output_team": "output_team",
            "FINISH": END
        }
    )
    
    # When a team finishes, go back to Astronomer to see if more is needed
    workflow.add_edge("knowledge_team", "astronomer")
    workflow.add_edge("data_team", "astronomer")
    workflow.add_edge("output_team", "astronomer")
    
    workflow.add_edge(START, "astronomer")
    
    return workflow.compile()

main_graph = build_main_graph()
