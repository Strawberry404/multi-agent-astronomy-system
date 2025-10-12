from langchain_core.messages import AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor

from state.state_definitions import KnowledgeTeamState
from config.config import Config

def web_search_node(state: KnowledgeTeamState, agent: AgentExecutor, name: str = "search") -> dict:
    """Web Search Agent - searches for recent information"""
    print(f"\n🔍 [WEB SEARCH] Executing search...")
    result = agent.invoke(state)
    output = result.get("output", "No output")
    print(f"[WEB SEARCH] Completed - Found: {output[:100]}...")
    return {
        "messages": [AIMessage(content=output, name=name)]
    }