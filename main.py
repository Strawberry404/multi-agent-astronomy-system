import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from graphs.main_graph import main_graph

# Suppress gRPC ALTS warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

load_dotenv()

def ask_astronomy_question(question: str):
    """Main entry point for the Astronomy Multi-Agent System."""
    print(f"\n🚀 Starting Astronomy Agent Session...")
    print(f"❓ Question: {question}\n")
    
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "next": "",
        "astronomical_data": {},
        "calculations": {},
        "visibility_info": {},
        "sources": [],
        "citations": [],
        "location": {"lat": 40.7128, "lon": -74.0060},
    }
    
    final_state = None
    try:
        for event in main_graph.stream(initial_state):
            for key, value in event.items():
                if "messages" in value and value["messages"]:
                    last_msg = value["messages"][-1]
                    print(f"🤖 [{key}]: {last_msg.content[:100]}...")
            final_state = event
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return None
        
    print("\n✅ Session Complete.")
    return final_state

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Astronomy Multi-Agent System")
    parser.add_argument("--query", type=str, help="Question to ask", default="Show me Mars photos")
    args = parser.parse_args()
    
    ask_astronomy_question(args.query)