import functools
from langgraph.graph import START, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

from config.config import Config
from state.state_definitions import KnowledgeTeamState
from agents.knowledge_team.rag_retriever import rag_retriever_node
from agents.knowledge_team.web_search import web_search_node
from agents.knowledge_team.citation_manager import citation_manager_node
from agents.supervisor import supervisor_node
from utils.helpers import create_agent


def build_knowledge_team():
    """Build the Knowledge Team orchestrator graph"""
    print("\n🔧 Building Knowledge Team graph...")
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL)
    
    # Create web search agent
    tavily_tool = TavilySearchResults(max_results=Config.TAVILY_MAX_RESULTS)
    search_agent = create_agent(
        llm,
        [tavily_tool],
        "You are a research assistant who can search for up-to-date astronomy information using the Tavily search engine."
    )
    
    # Create graph
    workflow = StateGraph(KnowledgeTeamState)
    
    # Add nodes
    workflow.add_node("supervisor", functools.partial(supervisor_node, llm=llm))
    workflow.add_node("rag_retriever", rag_retriever_node)
    workflow.add_node("search", functools.partial(web_search_node, agent=search_agent, name="search"))
    workflow.add_node("citation_manager", citation_manager_node)
    
    # Add conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "rag_retriever": "rag_retriever",
            "search": "search",
            "citation_manager": "citation_manager",
            "FINISH": END
        }
    )
    
    # Add edges back to supervisor
    workflow.add_edge("rag_retriever", "supervisor")
    workflow.add_edge("search", "supervisor")
    workflow.add_edge("citation_manager", "supervisor")
    
    # Start with supervisor
    workflow.add_edge(START, "supervisor")
    
    print("✓ Knowledge Team graph built")
    return workflow.compile()


# Compile the Knowledge Team
knowledge_team = build_knowledge_team()