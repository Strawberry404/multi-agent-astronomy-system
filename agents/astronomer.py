from typing import Dict, Any
from agents.supervisor import create_team_supervisor


def astronomer_agent_node(state: Dict[str, Any], llm) -> Dict[str, Any]:
    """Main orchestrator agent that routes queries to specialized teams."""
    teams = ["knowledge_team", "data_team", "output_team"]
    
    supervisor_chain = create_team_supervisor(
        llm,
        (
            "You are the Chief Astronomer, overseeing a multi-agent system. "
            "Route user queries to the most appropriate specialized team:\n"
            "- 'knowledge_team': For retrieving info from the PDF knowledge base or web search. (e.g. 'What is a black hole?', 'Latest news on SpaceX')\n"
            "- 'data_team': For specific NASA data, photos, calculations, sky positions. (e.g. 'Show me Mars photos', 'Distance to Andromeda', 'Is Jupiter visible?')\n"
            "- 'output_team': For creating educational explanations, observation plans, or visualizations/plots, EVEN IF the query involves data. (e.g. 'Explain this to a child', 'Plan my viewing night', 'Plot this data', 'Visualize planet distances')\n\n"
            "If a request requires multiple teams, route to the first one needed, and they will pass data along via the state. "
            "Respond FINISH when the user's request is fully satisfied."
        ),
        teams
    )
    
    result = supervisor_chain.invoke(state)
    next_step = result.get("next", "FINISH")
    return {"next": next_step}