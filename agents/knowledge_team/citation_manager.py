from langchain_core.messages import AIMessage
from state.state_definitions import KnowledgeTeamState


def citation_manager_node(state: KnowledgeTeamState) -> dict:
    """Citation Manager Agent - formats and tracks sources"""
    print("\n📋 [CITATION MANAGER] Extracting citations...")
    citations = []
    sources_found = state.get("sources", [])

    if sources_found:
        print(f"[CITATION MANAGER] Found {len(sources_found)} sources")
        for idx, source in enumerate(sources_found):
            citation = {
                "index": idx + 1,
                "source": source,
                "type": "PDF"
            }
            citations.append(citation)
            print(f"[CITATION MANAGER] [{idx+1}] {source}")
    else:
        print("[CITATION MANAGER] No sources found")

    # Format citations as a readable message
    if citations:
        formatted = "\n".join([
            f"[{c['index']}] {c['source']}"
            for c in citations
        ])
        citation_message = f"📚 Sources:\n{formatted}"
    else:
        citation_message = "No citations available"

    return {
        "messages": [AIMessage(content=citation_message, name="citation_manager")],
        "citations": citations
    }