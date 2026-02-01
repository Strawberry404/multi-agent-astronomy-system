from typing import Dict, Any
from langchain_core.messages import HumanMessage
from agents.supervisor import create_team_supervisor

def astronomer_agent_node(state: Dict[str, Any], llm) -> Dict[str, Any]:
    """Main orchestrator agent that routes queries to specialized teams."""
    teams = ["knowledge_team", "data_team", "output_team"]
    
    # --- INTELLIGENT ROUTING LOGIC (PREVENTS LOOPS) ---
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        
        # If Visualizer/Explainer finished, we are likely done.
        if hasattr(last_msg, 'name') and last_msg.name in ["visualizer", "explainer", "observation_planner"]:
             return {"next": "FINISH"}
             
        # If Data Team failed, route to Output Team to apologize/explain.
        if hasattr(last_msg, 'name') and "data_team" in str(last_msg.name).lower():
            if "error" in last_msg.content.lower() or "not found" in last_msg.content.lower():
                return {"next": "output_team"}
    # --------------------------------------------------

    supervisor_chain = create_team_supervisor(
        llm,
        (
            "You are the Chief Astronomer. Route user queries to the right team:\n"
            "- 'data_team': PRIMARY for fetching NEW data. Route here FIRST if the user asks for 'photos', 'images', 'distances', 'coordinates', or 'real-time positions'. (e.g. 'Show me a picture of Andromeda')\n"
            "- 'knowledge_team': For general definitions, history, or looking up facts in the PDF. (e.g. 'Who discovered Andromeda?')\n"
            "- 'output_team': For creating charts (PLOTS ONLY), detailed explanations, or viewing plans. Use this AFTER data is retrieved, or for pure educational questions.\n\n"
            "CRITICAL: If the user asks for a PHOTO, always route to 'data_team' first.\n"
            "Respond FINISH if the user's request has been fully satisfied."
        ),
        teams
    )
    
    result = supervisor_chain.invoke(state)
    next_step = result.get("next", "FINISH")
    return {"next": next_step}