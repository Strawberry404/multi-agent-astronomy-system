from typing import List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_google_genai import ChatGoogleGenerativeAI

from state.state_definitions import KnowledgeTeamState

"""each team will have a supervisor so this need to be generalized
afterwards , in a way where when we create the new team the members names are the ones
put into the logic invoked to it """
def create_team_supervisor(
    llm: ChatGoogleGenerativeAI,
    system_prompt: str,
    members: List[str]
):
    """Create supervisor for routing between team members"""
    options = ["FINISH"] + members

    function_def = {
        "name": "route",
        "description": "Select the next team member or finish.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "type": "string",
                    "title": "Next",
                    "enum": options,
                    "description": f"Select one of: {', '.join(options)}"
                }
            },
            "required": ["next"],
        }
    }

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", f"Given the conversation above, who should act next? Select one of: {options}")
    ])

    return (
        prompt
        | llm.bind(tools=[{"type": "function", "function": function_def}])
        | JsonOutputFunctionsParser()
    )


def supervisor_node(state: KnowledgeTeamState, llm: ChatGoogleGenerativeAI) -> dict:
    """Supervisor - routes tasks to appropriate team members"""
    print(f"\n👔 [SUPERVISOR] Analyzing routing decision...")
    print(f"[SUPERVISOR] Message history: {len(state['messages'])} messages")
    
    supervisor_chain = create_team_supervisor(
        llm,
        (
            "You are a supervisor managing a knowledge retrieval team. "
            "Your team specializes in gathering information from multiple sources:\n"
            "- 'rag_retriever': Searches the astronomy PDF knowledge base\n"
            "- 'search': Searches the web for recent information\n"
            "- 'citation_manager': Formats and tracks all sources\n\n"
            "For astronomy questions:\n"
            "1. First use 'rag_retriever' for foundational knowledge\n"
            "2. Optionally use 'search' for recent discoveries\n"
            "3. Always use 'citation_manager' to format sources\n"
            "4. Then FINISH\n\n"
            "Respond with FINISH when research is complete."
        ),
        state["team_members"]
    )
    
    result = supervisor_chain.invoke(state)
    next_agent = result.get("next", "FINISH") if isinstance(result, dict) else result
    print(f"[SUPERVISOR] Decision: Route to '{next_agent}'")
    
    return {"next": next_agent}