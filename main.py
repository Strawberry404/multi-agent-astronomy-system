from typing import List
from langchain_core.messages import HumanMessage
from agents.Managers.knowledge_team_graph import knowledge_team


def ask_astronomy_question(question: str, verbose: bool = True) -> dict:
    """
    Main interface to ask astronomy questions using the Knowledge Team
    
    Args:
        question: The astronomy question
        verbose: Print detailed execution trace
        
    Returns:
        Complete state including answer, sources, and citations
    """
    if verbose:
        print("=" * 80)
        print(f"🔭 ASTRONOMY QUESTION")
        print("=" * 80)
        print(f"Q: {question}\n")
    
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "team_members": ["rag_retriever", "search", "citation_manager"],
        "next": "",
        "sources": [],
        "citations": []
    }
    
    final_state = None
    for event in knowledge_team.stream(initial_state, stream_mode='values'):
        if event.get("messages"):
            latest = event["messages"][-1]
            if verbose:
                print(f"\n{'─' * 60}")
                print(f"Agent: {latest.name}")
                print(f"Output: {latest.content[:200]}...")
                print(f"{'─' * 60}")
        
        final_state = event
    
    if verbose:
        print("\n" + "=" * 80)
        print("📊 FINAL RESULTS")
        print("=" * 80)
        
        # Show the main answer (from RAG)
        for msg in final_state.get("messages", []):
            if msg.name == "rag_retriever":
                print(f"\n📝 Answer:\n{msg.content}\n")
        
        # Show citations
        if final_state.get("citations"):
            print(f"📚 Citations: {len(final_state['citations'])} sources")
            for citation in final_state["citations"]:
                print(f"  [{citation['index']}] {citation['source']}")
        
        print("=" * 80 + "\n")
    
    return final_state


def quick_ask(question: str) -> str:
    """Quick question - just returns the answer"""
    result = ask_astronomy_question(question, verbose=False)
    for msg in result.get("messages", []):
        if msg.name == "rag_retriever":
            return msg.content
    return "No answer generated"


def batch_questions(questions: List[str]) -> List[dict]:
    """Ask multiple questions in batch"""
    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n\n{'='*80}")
        print(f"QUESTION {i}/{len(questions)}")
        print(f"{'='*80}\n")
        result = ask_astronomy_question(q, verbose=True)
        results.append(result)
    return results


if __name__ == "__main__":
    print("\n" + "🌟" * 40)
    print("ASTRONOMY KNOWLEDGE TEAM - READY!")
    print("🌟" * 40)
    
    # Example usage
    result = ask_astronomy_question("what are the different galaxies known to human ")