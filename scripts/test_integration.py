import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import ask_astronomy_question

def test_integration():
    print("="*80)
    print("TEST: Integration - Multi-step Query")
    print("="*80)
    
    # Complex query requiring Data -> Output
    query = "Show me Mars rover photos and explain what they show."
    
    # Alternatively, a query requiring Knowledge -> Output
    # query = "What is a black hole? Explain it simply."
    
    print(f"Testing Query: {query}")
    ask_astronomy_question(query)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_integration()
