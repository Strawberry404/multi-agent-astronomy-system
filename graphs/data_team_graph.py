
import functools
from langgraph.graph import START, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI

from config.config import Config
from state.state_definitions import DataTeamState
from agents.data_team.database_agent import database_agent_node
from agents.data_team.calculator import calculator_node
from agents.data_team.sky_position import sky_position_node
from agents.supervisor import create_team_supervisor

def data_supervisor_node(state: DataTeamState, llm) -> dict:
    """Data Team Supervisor"""
    supervisor_chain = create_team_supervisor(
        llm,
        (
            "You are a data team supervisor. Route queries:\n"
            "- 'database_agent': For NASA data, object info\n"
            "- 'calculator': For calculations, conversions\n"
            "- 'sky_position': For visibility, positions\n"
            "Respond FINISH when done."
        ),
        state["team_members"]
    )
    
    result = supervisor_chain.invoke(state)
    next_agent = result.get("next", "FINISH")
    return {"next": next_agent}

def build_data_team():
    """Build Data Team graph"""
    llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL)

    workflow = StateGraph(DataTeamState)
    
    workflow.add_node("supervisor", functools.partial(data_supervisor_node, llm=llm))
    workflow.add_node("database_agent", database_agent_node)
    workflow.add_node("calculator", calculator_node)
    workflow.add_node("sky_position", sky_position_node)
    
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "database_agent": "database_agent",
            "calculator": "calculator",
            "sky_position": "sky_position",
            "FINISH": END
        }
    )
    
    workflow.add_edge("database_agent", "supervisor")
    workflow.add_edge("calculator", "supervisor")
    workflow.add_edge("sky_position", "supervisor")
    workflow.add_edge(START, "supervisor")
    
    return workflow.compile()

data_team = build_data_team()
