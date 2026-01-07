import functools
from langgraph.graph import START, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI

from config.config import Config
from agents.output_team.explainer import explainer_agent_node
from agents.output_team.observation_planner import observation_planner_node
from agents.output_team.visualizer import visualizer_agent_node
from agents.supervisor import create_team_supervisor

# Define a local state for output team if not using the global one
# For simplicity, we assume the passed state has necessary fields
# In main.py, we'll ensure state compatibility

def output_supervisor_node(state: dict, llm) -> dict:
    """Output Team Supervisor"""
    # Assuming 'team_members' key exists or we define it here
    members = ["explainer", "observation_planner", "visualizer"]
    
    supervisor_chain = create_team_supervisor(
        llm,
        (
            "You are the Output Team supervisor. Route queries:\n"
            "- 'explainer': For explanations, educational content\n"
            "- 'observation_planner': For viewing times, equipment, sky location\n"
            "- 'visualizer': For charts, plots, images\n"
            "Respond FINISH when done."
        ),
        members
    )
    
    result = supervisor_chain.invoke(state)
    next_agent = result.get("next", "FINISH")
    return {"next": next_agent}

def build_output_team():
    """Build Output Team graph"""
    llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL)

    workflow = StateGraph(dict) # Using dict for flexibility, or proper TypedDict
    
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
    
    workflow.add_edge("explainer", "supervisor")
    workflow.add_edge("observation_planner", "supervisor")
    workflow.add_edge("visualizer", "supervisor")
    workflow.add_edge(START, "supervisor")
    
    return workflow.compile()

output_team = build_output_team()
