from langchain_core.messages import HumanMessage
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graphs.output_team_graph import output_team

def test_output_team(question: str, context: dict = None):
    print("=" * 80)
    print(f"Q: {question}")
    print("=" * 80)
    
    initial_state = {
        "messages": [HumanMessage(content=question)],
        # We might need to initialize other state variables
    }
    if context:
        initial_state.update(context)
        
    try:
        for event in output_team.stream(initial_state, stream_mode='values'):
            if event.get("messages"):
                latest = event["messages"][-1]
                print(f"\nAgent: {latest.name}")
                print(f"Output: {latest.content[:500]}..." if len(latest.content) > 500 else f"Output: {latest.content}")
                print("\n")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test cases
    print("Test 1: Explainer")
    test_output_team("Explain what a black hole is to a 10 year old.")
    
    print("\nTest 2: Observation Planner")
    test_output_team("How can I see Saturn tonight?", context={"location": "New York"})
    
    print("\nTest 3: Visualizer")
    # This might fail if matplotlib backend is not configured for non-interactive or if dependencies aren't perfect, 
    # but let's try.
    test_output_team("Visualize the distances of planets in the solar system.")
