import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community.tools.tavily_search")

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor

from state.state_definitions import KnowledgeTeamState
from config.config import Config

# Initialize components globally or lazily
llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL)
search_tool = TavilySearchResults()
tools = [search_tool]

# Create a simple agent for search
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a web search assistant. Search for the user's query and provide a summary."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def web_search_node(state: KnowledgeTeamState) -> dict:
    """Web Search Agent - searches for recent information"""
    print(f"\n🔍 [WEB SEARCH] Executing search...")
    
    messages = state.get("messages", [])
    query = ""
    # Find the last user message
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            query = m.content
            break
            
    if not query:
        return {"messages": []}
    
    # Invoke the agent
    result = agent_executor.invoke({"input": query})
    output = result.get("output", "No output from search.")
    
    print(f"[WEB SEARCH] Completed - Found: {output[:100]}...")
    
    # We could also extract sources from result if available, but for now just returning content
    return {
        "messages": [AIMessage(content=output, name="web_search")]
    }


