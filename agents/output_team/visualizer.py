import os
import time
from typing import Dict, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from langchain_core.messages import AIMessage, HumanMessage
from config.config import Config


def visualizer_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Visualization Agent"""
    print("\n🎨 [VISUALIZER] Creating visualization...")
    
    messages = state.get("messages", [])
    query = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
             query = m.content.lower()
             break

    if not query:
        return {"messages": []}
    
    response_text = ""
    
    # Check if request is for a chart/plot
    if "chart" in query or "plot" in query or "visualize" in query:
        try:
            plt.figure(figsize=(10, 6))
            
            # Dummy data for demonstration
            if "planets" in query or "solar system" in query:
                objects = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter']
                distances = [0.4, 0.7, 1.0, 1.5, 5.2] # AU
                plt.bar(objects, distances, color='orange')
                plt.title("Distance from Sun (AU)")
                plt.ylabel("AU")
            else:
                # Default plot
                plt.plot([1, 2, 3], [1, 4, 9])
                plt.title("Sample Astronomical Data")
            
            # Save to file
            if not os.path.exists("static"):
                os.makedirs("static")
                
            filename = f"plot_{int(time.time())}.png"
            filepath = os.path.join("static", filename)
            plt.savefig(filepath)
            plt.close()
            
            response_text = f"I've generated a visualization for you:\n![Visualization]({filepath})"
            
        except Exception as e:
            response_text = f"❌ Failed to generate visualization: {e}"
    else:
        response_text = "I can create charts and visualizations. Ask me to visualize planet distances or star charts."

    return {
        "messages": [AIMessage(content=response_text, name="visualizer")]
    }
