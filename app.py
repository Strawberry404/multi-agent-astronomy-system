import streamlit as st
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Add project root to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from graphs.main_graph import main_graph

# Page config
st.set_page_config(page_title="Astronomy Multi-Agent System", page_icon="🚀", layout="wide")

# Title and sidebar
st.title("🚀 Astronomy Multi-Agent System")
st.markdown("### Powered by LangGraph, Google Gemini, and NASA APIs")

with st.sidebar:
    st.header("About")
    st.markdown("""
    This system uses three specialized teams to answer your questions:
    
    1. **Knowledge Team**: Searchesthe PDF knowledge base and web.
    2. **Data Team**: Fetches real-time data from NASA APIs (Photos, Objects) and performs calculations.
    3. **Output Team**: Explains concepts, plans observations, and creates visualizations.
    """)
    st.divider()
    location_lat = st.number_input("Latitude", value=40.7128)
    location_lon = st.number_input("Longitude", value=-74.0060)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask an astronomy question... (e.g., 'Show me Mars photos')"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # placeholder for thinking
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Initial State
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "next": "",
            "astronomical_data": {},
            "calculations": {},
            "visibility_info": {},
            "sources": [],
            "citations": [],
            "location": {"lat": location_lat, "lon": location_lon},
        }

        # Run graph
        status_container = st.status("Agents working...", expanded=True)
        
        try:
            for event in main_graph.stream(initial_state):
                for key, value in event.items():
                    # Update status
                    if "messages" in value and value["messages"]:
                        last_msg = value["messages"][-1]
                        agent_name = key.replace("_", " ").title()
                        
                        # Show what each agent is doing
                        status_container.write(f"**{agent_name}**: {last_msg.content[:100]}...")
                        
                        # If output team produced something, accumulate it
                        # But typically the final answer comes from the last step or we can just append
                        # Here we just look at the final output or specific interesting updates
                        
                        # Determine if this is a 'final' type message to show in the main chat
                        # Usually the Explainer or final agent content is what we want.
                        # For now, we'll store the last message content as the response
                        if hasattr(last_msg, 'content'):
                            full_response = last_msg.content
            
            status_container.update(label="Complete!", state="complete", expanded=False)
            
            # Display response
            message_placeholder.markdown(full_response)
            
            # Use regex or check for image paths in the response to display actual images if markdown didn't catch them
            # (Streamlit markdown handles local images if path is correct relative to app, 
            # but usually st.image is better. For now relying on markdown ![]() syntax which st.markdown supports)
            
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            status_container.update(label="Error", state="error")
            st.error(f"Error: {str(e)}")
