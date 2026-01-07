from langchain_core.messages import HumanMessage
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graphs.data_team_graph import data_team

def test_data_team(question: str):
    print("=" * 80)
    print(f"Q: {question}")
    print("=" * 80)
    
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "team_members": ["database_agent", "calculator", "sky_position"],
        "next": "",
        "astronomical_data": {},
        "calculations": {},
        "visibility_info": {},
        "location": {"lat": 40.7128, "lon": -74.0060},
        "date_time": None
    }
    
    try:
        for event in data_team.stream(initial_state, stream_mode='values'):
            if event.get("messages"):
                latest = event["messages"][-1]
                print(f"\nAgent: {latest.name}")
                print(f"Output: {latest.content}\n")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test cases
    test_data_team("Tell me about Jupiter")
    test_data_team("Convert 4.2 light years to kilometers")
    test_data_team("When does Mars rise tonight?")
    test_data_team("Show me Mars rover photos")
    # test_data_team("How many exoplanets have we found?") # Optional, can be slow
