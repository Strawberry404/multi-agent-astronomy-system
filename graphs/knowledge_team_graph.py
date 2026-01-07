import functools
from langgraph.graph import START, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import Config
from state.state_definitions import KnowledgeTeamState
from agents.knowledge_team.rag_retriever import rag_retriever_node
from agents.knowledge_team.web_search import web_search_node
from agents.knowledge_team.citation_manager import citation_manager_node
from agents.supervisor import create_team_supervisor

def knowledge_supervisor_node(state: KnowledgeTeamState, llm) -> dict:
    """Knowledge Team Supervisor"""
    members = ["rag_retriever", "web_search", "citation_manager"]
    
    # Check if we already have responses from agents
    messages = state.get("messages", [])
    agent_responses = [msg for msg in messages if hasattr(msg, 'name') and msg.name in members]
    
    # If we have responses and citations are done, finish
    if agent_responses and state.get("citations"):
        return {"next": "FINISH"}
    
    supervisor_chain = create_team_supervisor(
        llm,
        (
            "You are the Knowledge Team supervisor. Route queries:\n"
            "- 'rag_retriever': For information from the internal knowledge base (PDF).\n"
            "- 'web_search': For recent events, news, or info not in the PDF.\n"
            "- 'citation_manager': To format citations and sources (only after rag_retriever or web_search has responded).\n"
            "IMPORTANT: After getting information and citations, respond FINISH. Do not loop."
        ),
        members
    )
    
    result = supervisor_chain.invoke(state)
    next_agent = result.get("next", "FINISH")
    return {"next": next_agent}

def build_knowledge_team():
    """Build Knowledge Team graph"""
    llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL)

    workflow = StateGraph(KnowledgeTeamState)
    
    workflow.add_node("supervisor", functools.partial(knowledge_supervisor_node, llm=llm))
    workflow.add_node("rag_retriever", rag_retriever_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("citation_manager", citation_manager_node)
    
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "rag_retriever": "rag_retriever",
            "web_search": "web_search",
            "citation_manager": "citation_manager",
            "FINISH": END
        }
    )
    
    workflow.add_edge("rag_retriever", "supervisor")
    workflow.add_edge("web_search", "supervisor")
    workflow.add_edge("citation_manager", "supervisor")
    workflow.add_edge(START, "supervisor")
    
    return workflow.compile()

knowledge_team = build_knowledge_team()
