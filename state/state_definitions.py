from typing import TypedDict , List , Annotated , Dict , Any
import operator
from langchain_core.messages import BaseMessage
from typing import TypedDict, List, Optional
from langchain_core.documents import Document

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


class DataTeamState(TypedDict):
    """State for the knowledge team subgroup"""
    messages:Annotated[list[BaseMessage], operator.add]
    team_members:List[str]
    next:str

    astronomical_data: Dict[str, Any]  # From database agent
    calculations: Dict[str, Any]       # From calculator agent
    visibility_info: Dict[str, Any]  
    

    # User context
    location: Optional[Dict[str, float]]  # {"lat": 40.7, "lon": -74.0}
    date_time: Optional[str]  # ISO format or "now"



# we will be adding other states the goal is for 
# centralization and keeping on the clean code


# the centralized one:


class AstronomyState(TypedDict):
    # Input
    question: str
    user_location: Optional[dict]
    
    # Knowledge Team
    rag_context: List[Document]
    web_results: List[dict]
    citations: List[str]
    
    # Data Team
    astronomical_data: dict
    calculations: dict
    visibility_info: dict
    
    # Output Team
    explanation: str
    observation_plan: str
    visualizations: List[str]
    
    # Final
    answer: str
    metadata: dict