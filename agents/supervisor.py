from typing import List, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputToolsParser
from langchain_google_genai import ChatGoogleGenerativeAI

def create_team_supervisor(llm: ChatGoogleGenerativeAI, system_prompt: str, members: List[str]) -> Any:
    """An LLM-based supervisor."""
    
    options = ["FINISH"] + members
    
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next Role",
                    "type": "string",
                    "enum": options
                },
            },
            "required": ["next"],
        },
    }
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next? "
                "Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    
    return (
        prompt
        | llm.bind_tools(tools=[function_def], tool_choice="route")
        | JsonOutputToolsParser() 
        | (lambda x: x[0]["args"])
    )
