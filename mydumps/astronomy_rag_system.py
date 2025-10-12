# ========================================
# File: astronomy_rag_system.py
# Complete RAG System for Astronomy Agent
# ========================================

from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import hub


# ========================================
# 1. SETUP & CONFIGURATION
# ========================================

class AstronomyRAGConfig:
    """Configuration for the RAG system"""
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL = "gemini-2.5-flash"
    VECTOR_STORE_PATH = "faiss_index"
    RETRIEVAL_K = 10  # Number of documents to retrieve
    DEVICE = "cuda"  # or "cpu" if no GPU


# ========================================
# 2. STATE DEFINITION
# ========================================

class AstronomyRAGState(TypedDict):
    """State for the astronomy RAG system"""
    question: str
    context: List[Document]
    answer: str
    metadata: dict  # Additional info like sources, confidence, etc.


# ========================================
# 3. INITIALIZE COMPONENTS
# ========================================

def initialize_rag_components():
    """Initialize embedding model, LLM, and vector store"""
    print("🚀 Initializing RAG Components...")
    
    # Embedding model
    embed_model = HuggingFaceEmbeddings(
        model_name=AstronomyRAGConfig.EMBEDDING_MODEL,
        model_kwargs={"device": AstronomyRAGConfig.DEVICE}
    )
    print(f"✓ Loaded embedding model: {AstronomyRAGConfig.EMBEDDING_MODEL}")
    
    # LLM
    llm = ChatGoogleGenerativeAI(model=AstronomyRAGConfig.LLM_MODEL)
    print(f"✓ Loaded LLM: {AstronomyRAGConfig.LLM_MODEL}")
    
    # Vector store
    vector_store = FAISS.load_local(
        AstronomyRAGConfig.VECTOR_STORE_PATH,
        embed_model,
        allow_dangerous_deserialization=True
    )
    print(f"✓ Loaded vector store from: {AstronomyRAGConfig.VECTOR_STORE_PATH}")
    
    # RAG prompt
    prompt = hub.pull('rlm/rag-prompt')
    print("✓ Loaded RAG prompt template")
    
    return embed_model, llm, vector_store, prompt


# Initialize global components
embed_model, llm, vector_store, prompt = initialize_rag_components()


# ========================================
# 4. RAG NODES
# ========================================

def retrieve(state: AstronomyRAGState) -> dict:
    """Retrieve relevant documents from vector store"""
    print("\n📚 --Retrieving Information from Vector Store--")
    question = state['question']
    
    # Retrieve documents with scores
    retrieved_docs = vector_store.similarity_search_with_score(
        query=question,
        k=AstronomyRAGConfig.RETRIEVAL_K
    )
    
    # Separate docs and scores
    docs = [doc for doc, score in retrieved_docs]
    scores = [score for doc, score in retrieved_docs]
    
    print(f"✓ Retrieved {len(docs)} documents")
    
    # Show top 3 results
    for i, (doc, score) in enumerate(zip(docs[:3], scores[:3])):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        print(f"  Doc {i+1}: {source} (Page {page}) - Similarity: {score:.4f}")
        print(f"    Preview: {doc.page_content[:100]}...")
    
    # Prepare metadata
    metadata = {
        'num_docs_retrieved': len(docs),
        'sources': [
            {
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'N/A'),
                'score': float(score)
            }
            for doc, score in zip(docs, scores)
        ]
    }
    
    return {
        'context': docs,
        'metadata': metadata
    }


def generate(state: AstronomyRAGState) -> dict:
    """Generate answer using retrieved context"""
    print("\n🤖 --Generating Response from RAG Context--")
    
    # Combine document contents
    docs_content = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in state['context']
    ])
    
    # Generate response
    messages = prompt.invoke({
        'question': state['question'],
        'context': docs_content
    })
    response = llm.invoke(messages)
    
    answer = response.content if hasattr(response, 'content') else str(response)
    
    print("✓ Response Generated Successfully")
    print(f"  Answer length: {len(answer)} characters")
    
    # Update metadata
    metadata = state.get('metadata', {})
    metadata['answer_length'] = len(answer)
    
    return {
        'answer': answer,
        'metadata': metadata
    }


# ========================================
# 5. BUILD GRAPH
# ========================================

def build_rag_graph():
    """Build and compile the RAG graph"""
    print("\n🔧 Building RAG Graph...")
    
    rag_graph = StateGraph(AstronomyRAGState)
    
    # Add nodes
    rag_graph.add_node("retrieve", retrieve)
    rag_graph.add_node("generate", generate)
    
    # Add edges
    rag_graph.add_edge(START, "retrieve")
    rag_graph.add_edge("retrieve", "generate")
    rag_graph.add_edge("generate", END)
    
    # Compile
    rag_agent = rag_graph.compile()
    
    print("✓ RAG Graph compiled successfully")
    return rag_agent


# Initialize the agent
rag_agent = build_rag_graph()


# ========================================
# 6. USAGE FUNCTIONS
# ========================================

def ask_astronomy_question(question: str, verbose: bool = True) -> dict:
    """
    Main function to ask astronomy questions
    
    Args:
        question: The astronomy question to ask
        verbose: Whether to print detailed progress
        
    Returns:
        dict with 'answer', 'context', and 'metadata'
    """
    if verbose:
        print("=" * 80)
        print(f"🔭 ASTRONOMY QUESTION: {question}")
        print("=" * 80)
    
    # Run the RAG agent
    result = rag_agent.invoke({
        'question': question,
        'context': [],
        'answer': '',
        'metadata': {}
    })
    
    if verbose:
        print("\n" + "=" * 80)
        print("📝 ANSWER:")
        print("=" * 80)
        print(result['answer'])
        print("\n" + "=" * 80)
        print("📊 METADATA:")
        print("=" * 80)
        print(f"  Documents retrieved: {result['metadata']['num_docs_retrieved']}")
        print(f"  Answer length: {result['metadata']['answer_length']} characters")
        print(f"\n  Top 3 Sources:")
        for i, source in enumerate(result['metadata']['sources'][:3], 1):
            print(f"    {i}. {source['source']} (Page {source['page']}) - Score: {source['score']:.4f}")
        print("=" * 80)
    
    return result


def batch_ask_questions(questions: List[str]) -> List[dict]:
    """Ask multiple questions in batch"""
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n\n{'='*80}")
        print(f"QUESTION {i}/{len(questions)}")
        print(f"{'='*80}")
        result = ask_astronomy_question(question, verbose=True)
        results.append(result)
    return results


def get_relevant_sources(question: str, k: int = 5) -> List[dict]:
    """
    Get only the relevant sources without generating an answer
    Useful for exploring what documents are available
    """
    print(f"🔍 Finding relevant sources for: {question}")
    
    retrieved_docs = vector_store.similarity_search_with_score(
        query=question,
        k=k
    )
    
    sources = []
    for i, (doc, score) in enumerate(retrieved_docs, 1):
        source_info = {
            'rank': i,
            'source': doc.metadata.get('source', 'Unknown'),
            'page': doc.metadata.get('page', 'N/A'),
            'similarity_score': float(score),
            'preview': doc.page_content[:200] + "..."
        }
        sources.append(source_info)
        
        print(f"\n{i}. {source_info['source']} (Page {source_info['page']})")
        print(f"   Similarity: {score:.4f}")
        print(f"   Preview: {source_info['preview']}")
    
    return sources


# ========================================
# 7. EXAMPLE USAGE
# ========================================

if __name__ == "__main__":
    print("\n" + "🌟" * 40)
    print("ASTRONOMY RAG SYSTEM - READY!")
    print("🌟" * 40)
    
    # Example 1: Single question
    print("\n\n📌 EXAMPLE 1: Single Question")
    result = ask_astronomy_question(
        "What is a black hole and how does it form?",
        verbose=True
    )
    
    # Example 2: Just get sources (no answer generation)
    print("\n\n📌 EXAMPLE 2: Find Relevant Sources Only")
    sources = get_relevant_sources(
        "stellar evolution",
        k=3
    )
    
    # Example 3: Multiple questions
    print("\n\n📌 EXAMPLE 3: Batch Questions")
    questions = [
        "What is the life cycle of a star?",
        "Explain the difference between planets and dwarf planets",
        "What causes the seasons on Earth?"
    ]
    batch_results = batch_ask_questions(questions)
    
    # Example 4: Access specific parts of the result
    print("\n\n📌 EXAMPLE 4: Accessing Result Components")
    result = ask_astronomy_question("What is a nebula?", verbose=False)
    
    print("Answer:", result['answer'][:200] + "...")
    print("\nNumber of context documents:", len(result['context']))
    print("First document preview:", result['context'][0].page_content[:100])
    print("\nMetadata:", result['metadata'])


# ========================================
# 8. QUICK ACCESS FUNCTIONS FOR NOTEBOOK
# ========================================

def quick_ask(question: str) -> str:
    """Quick function for notebook use - just returns the answer"""
    result = rag_agent.invoke({
        'question': question,
        'context': [],
        'answer': '',
        'metadata': {}
    })
    return result['answer']


def ask_with_sources(question: str) -> tuple:
    """Returns both answer and sources"""
    result = ask_astronomy_question(question, verbose=False)
    sources = [
        f"{s['source']} (Page {s['page']})"
        for s in result['metadata']['sources'][:3]
    ]
    return result['answer'], sources


# ========================================
# USAGE EXAMPLES IN NOTEBOOK:
# ========================================
"""
# Simple usage:
answer = quick_ask("What is a supernova?")
print(answer)

# With sources:
answer, sources = ask_with_sources("Explain gravitational waves")
print(answer)
print("\nSources:", sources)

# Full details:
result = ask_astronomy_question("What is dark matter?")

# Just exploring sources:
sources = get_relevant_sources("quantum mechanics in space", k=5)
"""