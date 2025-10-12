from typing import TypedDict , List , Annotated
import operator
from langchain_core.messages import BaseMessage


class KnowledgeTeamState(TypedDict):
    """State for the knowledge team subgroup"""
    messages:Annotated[list[BaseMessage], operator.add]
    team_members:List[str]
    next:str
    sources:List[str]
    citations:List[dict]

class RAGState(TypedDict):
    """Internal state for RAG Processing"""
    question:str
    context:List
    answer:str

# we will be adding other states the goal is for 
# centralization and keeping on the clean code