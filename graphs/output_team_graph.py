import functools
from langgraph.graph import START, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI

from config.config import Config
from agents.output_team.explainer import explainer_agent_node
from agents.output_team.observation_planner import observation_planner_node
from agents.output_team.visualizer import visualizer_agent_node
from agents.supervisor import create_team_supervisor
from state.state_definitions import OutputTeamState  # <--- Import the new state

from typing import Dict, Any

def output_supervisor_node(state: OutputTeamState, llm) -> Dict[str, Any]:
    """Output Team Supervisor"""
    members = ["explainer", "observation_planner", "visualizer"]
    
    # Check if any agent has already responded
    messages = state.get("messages", [])
    
    # Look for the LAST message. If it's from a team member, we are done.
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, 'name') and last_msg.name in members:
            return {"next": "FINISH"}
    
    supervisor_chain = create_team_supervisor(
        llm,
        (
            "You are the Output Team supervisor. Route queries:\n"
            "- 'explainer': For explanations, educational content\n"
            "- 'observation_planner': For viewing times, equipment, sky location\n"
            "- 'visualizer': For charts, plots, images\n"
            "IMPORTANT: If the last message is from one of your agents, respond with FINISH."
        ),
        members
    )
    
    result = supervisor_chain.invoke(state)
    next_agent = result.get("next", "FINISH")
    return {"next": next_agent}

def build_output_team():
    """Build Output Team graph"""
    llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL)

    # Use OutputTeamState instead of dict
    workflow = StateGraph(OutputTeamState) 
    
    workflow.add_node("supervisor", functools.partial(output_supervisor_node, llm=llm))
    workflow.add_node("explainer", explainer_agent_node)
    workflow.add_node("observation_planner", observation_planner_node)
    workflow.add_node("visualizer", visualizer_agent_node)
    
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next"),
        {
            "explainer": "explainer",
            "observation_planner": "observation_planner",
            "visualizer": "visualizer",
            "FINISH": END
        }
    )
    
    # Important: Agents loop back to supervisor to check "FINISH" condition
    workflow.add_edge("explainer", "supervisor")
    workflow.add_edge("observation_planner", "supervisor")
    workflow.add_edge("visualizer", "supervisor")
    workflow.add_edge(START, "supervisor")
    
    return workflow.compile()

output_team = build_output_team()