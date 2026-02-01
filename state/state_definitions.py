from typing import TypedDict, List, Annotated, Dict, Any, Optional
import operator
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


class KnowledgeTeamState(TypedDict, total=False):
    """State for the knowledge team subgroup."""
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str
    sources: List[str]
    citations: List[dict]


class RAGState(TypedDict):
    """Internal state for RAG processing."""
    question: str
    context: List[Document]
    answer: str


class DataTeamState(TypedDict, total=False):
    """State for the data team subgroup."""
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str
    astronomical_data: Dict[str, Any]
    calculations: Dict[str, Any]
    visibility_info: Dict[str, Any]
    location: Optional[Dict[str, float]]
    date_time: Optional[str]


# --- ADD THIS NEW CLASS ---
class OutputTeamState(TypedDict, total=False):
    """State for the output team subgroup."""
    messages: Annotated[List[BaseMessage], operator.add] # <--- Crucial: Appends messages!
    team_members: List[str]
    next: str
    astronomical_data: Dict[str, Any] # Inherit data from other teams
    calculations: Dict[str, Any]