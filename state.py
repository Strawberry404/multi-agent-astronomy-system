from langchain.document_loaders import PyPDFLoader

filename = "/content/drive/MyDrive/Astronomy.pdf"
loader = PyPDFLoader(filename)
docs = [ ]
docs = loader.load()# Call the load method to get the documents
print(docs)
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,  # Much larger chunks
    chunk_overlap=500,
    add_start_index=True
    )

all_splits = text_splitter.split_documents(docs)

print(f"Split the pdf into {len(all_splits)} sub-documents")
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"  ,
    model_kwargs={"device": "cuda"}  )

vector_store = FAISS.from_documents(
    documents=all_splits,
    embedding=embed_model
)

# Save locally
vector_store.save_local("faiss_index")
from typing import TypedDict, List, Annotated, Union
import functools
import operator
from IPython.display import Image, display

from langchain import hub
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser

from langgraph.graph import START, END, StateGraph
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings


embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"  ,
    model_kwargs={"device": "cuda"}  )

prompt = hub.pull('rlm/rag-prompt')


llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

vector_store = FAISS.load_local("faiss_index", embed_model, allow_dangerous_deserialization=True)


class RAGState(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: RAGState):
    print("--Retrieving Information from Vector Store--")
    question = state['question']
    retrieved_docs = vector_store.similarity_search(query=question, k=10)
    print(f"✓ RAG Retrieved {len(retrieved_docs)} documents")
    for i, doc in enumerate(retrieved_docs[:3]):
        print(f"  Doc {i+1}: {doc.metadata.get('source', 'Unknown')} (Score: {doc.metadata.get('score', 'N/A')})")
    return {'context': retrieved_docs}

def generate(state: RAGState):
    print("→ Generating Response from RAG Context")
    docs_content = "\n\n".join([doc.page_content for doc in state['context']])
    messages = prompt.invoke({'question': state['question'], 'context': docs_content})
    response = llm.invoke(messages)
    print("✓ RAG Response Generated Successfully")
    return {'answer': response.content if hasattr(response, 'content') else str(response)}

rag_graph = StateGraph(RAGState)
rag_graph.add_node("retrieve", retrieve)
rag_graph.add_node("generate", generate)
rag_graph.add_edge(START, "retrieve")
rag_graph.add_edge("retrieve", "generate")
rag_graph.add_edge("generate", END)
rag_agent = rag_graph.compile()


# Generate and display the visualization
try:
    display(Image(rag_agent.get_graph().draw_mermaid_png()))
except Exception:
    # Fallback if mermaid rendering fails
    print(rag_agent.get_graph().draw_ascii())

for event in rag_agent.stream({'question': 'What is a neutron star?'},stream_mode='values'):
if event.get('answer',''):
    print(event['answer'])


class ResearchTeamState(TypedDict):
  messages: Annotated[list[BaseMessage] , operator.add]
  team_members:List[str]
  next:str


def agent_node(state: ResearchTeamState, agent: AgentExecutor, name: str):
    """Execute an agent and format the result"""
    print(f"\n[AGENT] Executing {name.upper()} agent")
    result = agent.invoke(state)
    output = result.get("output", "No output")
    print(f"[AGENT] {name.upper()} completed - Output: {output[:100]}...")
    return {
        "messages": [AIMessage(content=output, name=name)]
    }

def supervisor_node(state: ResearchTeamState):
    """Supervisor routing logic"""
    print(f"\n[SUPERVISOR] Routing decision needed")
    print(f"[SUPERVISOR] Current message history: {len(state['messages'])} messages")
    supervisor_chain = create_team_supervisor(
        llm,
        (
            "You are a supervisor managing research agents. "
            "Available team members can perform specialized research tasks. "
            "Route tasks to appropriate team members based on the query. "
            "Respond with FINISH when research is complete."
        ),
        state["team_members"]
    )
    result = supervisor_chain.invoke(state)
    next_agent = result.get("next", "FINISH") if isinstance(result, dict) else result
    print(f"[SUPERVISOR] Routing to: {next_agent}")
    return {"next": next_agent}


def rag_agent_node(state: ResearchTeamState):
    """RAG agent node - processes questions through the RAG system"""
    print("\n[RAG AGENT] Processing query through RAG system...")
    if state["messages"]:
        last_message = state["messages"][-1].content
        print(f"[RAG AGENT] Query: {last_message}")
        rag_result = rag_agent.invoke({"question": last_message})
        answer = rag_result.get("answer", "No answer generated")
        context = rag_result.get("context", [])
        print(f"[RAG AGENT] Answer generated with {len(context)} context documents")
        return {
            "messages": [AIMessage(content=answer, name="rag_retriever")],
            "sources": [doc.metadata.get("source", "Unknown") for doc in context]
        }
    return {"messages": []}


def create_agent(
    llm:ChatGoogleGenerativeAI,
    tools : list ,
    system_prompt:str) -> AgentExecutor:
    """Create a function-calling agent and add it to the graph"""
    system_prompt+=("\nWork autonomously according to your specialty , using the tools available to you."
    "Do not ask for clarification"
    "your other team members (and other teams) will collaborate with you with their own specialties"
    "You are chosen for a reason! you are one of the following team members:{team_members}"
    )
    prompt=ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]

    )
    agent = create_tool_calling_agent(llm , tools , prompt)
    executor = AgentExecutor(agent=agent,tools = tools)
    return executor


from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.runnables import bind_functions # Remove this import
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser # Import the correct parser

def create_team_supervisor(
    llm: ChatGoogleGenerativeAI,
    system_prompt: str,
    members: List[str]
):
    """Create an LLM-based router/supervisor"""
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


      | JsonOutputFunctionsParser() # Use the correct parser

  )



def citation_manager_node(state: ResearchTeamState):
    """Extract and format citations from RAG results"""
    print("\n[CITATION MANAGER] Extracting citations...")
    citations = []
    sources_found = state.get("sources", [])

    if sources_found:
        print(f"[CITATION MANAGER] Found {len(sources_found)} sources")
        for source in sources_found:
            citation = {
                "source": source,
                "page": state["messages"][-1].additional_kwargs.get("page", "N/A"),
                "timestamp": str(state["messages"][-1].response_metadata.get("created_at", ""))
            }
            citations.append(citation)
            print(f"[CITATION MANAGER] - {source}")
    else:
        print("[CITATION MANAGER] No sources found in state")

    # Format citations for display
    formatted_citations = "\n".join([
        f"[{i+1}] {c['source']} (Page: {c['page']})"
        for i, c in enumerate(citations)
    ])

    citation_message = f"📚 Citations:\n{formatted_citations}" if citations else "No citations available"
    print(f"[CITATION MANAGER] Formatted citations: {citation_message[:100]}...")

    return {
        "messages": [AIMessage(content=citation_message, name="citation_manager")],
        "citations": citations
    }



tavily_tool = TavilySearchResults(max_results =5)

search_agent = create_agent(
    llm ,
    [tavily_tool],
    "You are a research Assitant who can search for up-to-date info using the tavily search engine ."
)

search_node = functools.partial(agent_node , agent=search_agent , name="search")



workfloaw_1 = StateGraph(ResearchTeamState)
workfloaw_1.add_node("supervisor" , supervisor_node)
workfloaw_1.add_node("search" , functools.partial(agent_node, agent=search_agent, name="search"))
workfloaw_1.add_node("rag_retriever" , rag_agent_node)
workfloaw_1.add_node("citation_manager" , citation_manager_node)

workfloaw_1.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "search": "search",
        "rag_retriever": "rag_retriever",
        "citation_manager": "citation_manager",
        "FINISH": END
    }
)


workfloaw_1.add_edge("search", "supervisor")
workfloaw_1.add_edge("rag_retriever", "supervisor")
workfloaw_1.add_edge("citation_manager", "supervisor")
workfloaw_1.add_edge(START, "supervisor")

Knowledge_Team_orchestrator = workfloaw_1.compile()

if __name__ == "__main__":
    print("="*60)
    print("MULTI-AGENT ORCHESTRATOR WITH RAG + CITATIONS")
    print("="*60)

    initial_state = {
        "messages": [HumanMessage(content="What is a neutron star?")],
        "team_members": ["search", "rag_retriever", "citation_manager"],
        "next": "",
        "sources": []
    }


    print(f"\n[START] Query: {initial_state['messages'][0].content}\n")

    for event in Knowledge_Team_orchestrator.stream(initial_state, stream_mode='values'):
        if event.get("messages"):
            latest = event["messages"][-1]
            print(f"\n{'─'*60}")
            print(f"Agent: {latest.name}")
            print(f"Message: {latest.content}")
            print(f"{'─'*60}")

        if event.get("citations"):
            print(f"\n[CITATIONS] {event['citations']}")

    print("\n" + "="*60)
    print("ORCHESTRATION COMPLETE")
    print("="*60)