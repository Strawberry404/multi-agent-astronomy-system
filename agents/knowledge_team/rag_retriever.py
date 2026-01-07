from langchain import hub
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph

from config.config import Config
from state.state_definitions import KnowledgeTeamState, RAGState

# Initialize components
embed_model = HuggingFaceEmbeddings(
    model_name=Config.EMBEDDING_MODEL,
    model_kwargs={"device":Config.DEVICE}  
)

vector_store = FAISS.load_local(
    Config.VECTOR_STORE_PATH, 
    embed_model, 
    allow_dangerous_deserialization=True
)

llm = ChatGoogleGenerativeAI(model = Config.LLM_MODEL)
rag_prompt = hub.pull('rlm/rag-prompt')


def rag_retrieve(state: RAGState) -> dict:
    """Retrieve documents from vector store"""
    print("  [RAG] Retrieving from vector store...")
    question = state['question']
    retrieved_docs = vector_store.similarity_search(query=question, k=Config.RAG_K)
    print(f"  [RAG] Retrieved {len(retrieved_docs)} documents")
    return {'context': retrieved_docs}

def rag_generate(state: RAGState) -> dict:
    """Generate answer from retrieved context"""
    print("  [RAG] Generating response...")
    docs_content = "\n\n".join([doc.page_content for doc in state['context']])
    messages = rag_prompt.invoke({
        'question': state['question'],
        'context': docs_content
    })
    response = llm.invoke(messages)
    print("  [RAG] Response generated")
    return {'answer': response.content if hasattr(response, 'content') else str(response)}

#Build Rag sub-graph

rag_graph = StateGraph(RAGState)
rag_graph.add_node("retrieve", rag_retrieve)
rag_graph.add_node("generate", rag_generate)
rag_graph.add_edge(START, "retrieve")
rag_graph.add_edge("retrieve", "generate")
rag_graph.add_edge("generate", END)
rag_agent = rag_graph.compile()


# Main RAG retriever node for Knowledge Team
def rag_retriever_node(state: KnowledgeTeamState) -> dict:
    """RAG Retriever Agent - queries the PDF knowledge base"""
    print("\n📚 [RAG RETRIEVER] Processing query...")
    
    messages = state.get("messages", [])
    query = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
             query = m.content
             break
             
    if query:
        last_message = query
        print(f"[RAG RETRIEVER] Query: {last_message}")
        
        # Invoke the RAG sub-agent
        rag_result = rag_agent.invoke({"question": last_message})
        answer = rag_result.get("answer", "No answer generated")
        context = rag_result.get("context", [])
        
        # Extract sources for citation manager
        sources = [doc.metadata.get("source", "Unknown") for doc in context]
        
        print(f"[RAG RETRIEVER] Generated answer with {len(context)} source documents")
        
        return {
            "messages": [AIMessage(content=answer, name="rag_retriever")],
            "sources": sources
        }
    
    return {"messages": []}