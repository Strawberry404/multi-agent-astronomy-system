
#this is still not final

from state.state_definitions import AstronomyState
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import RAGConfig

llm = ChatGoogleGenerativeAI(model=RAGConfig.LLM_MODEL)

def astronomer_node(state: AstronomyState) -> dict:
    """Main orchestrator - analyzes query and prepares for routing"""
    print("🔭 Astronomer analyzing query...")
    
    question = state['question']
    
    # Simple analysis for now
    query_info = {
        'needs_rag': True,  # Always use RAG for now
        'needs_realtime': False,  # Phase 4
        'needs_calculation': False,  # Phase 4
    }
    
    metadata = state.get('metadata', {})
    metadata['query_analysis'] = query_info
    
    return {'metadata': metadata}

def generate_answer_node(state: AstronomyState) -> dict:
    """Generate final answer from all gathered context"""
    print("💭 Generating final answer...")
    
    # Combine RAG context
    docs_content = "\n\n".join([
        doc.page_content for doc in state.get('rag_context', [])
    ])
    
    prompt = f"""You are an expert astronomer. Answer the following question using the provided context.

Question: {state['question']}

Context:
{docs_content}

Provide a clear, accurate, and educational answer."""

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)
    
    return {'answer': answer}