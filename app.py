import os
import re
import traceback
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Suppress gRPC ALTS warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

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
    
    1. **Knowledge Team**: Searches the PDF knowledge base and web.
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

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
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

        status_container = st.status("Agents working...", expanded=True)
        final_state = {}
        
        try:
            for event in main_graph.stream(initial_state):
                for key, value in event.items():
                    final_state.update(value)
                    if "messages" in value and value["messages"]:
                        last_msg = value["messages"][-1]
                        agent_name = key.replace("_", " ").title()
                        status_container.write(f"**{agent_name}**: {last_msg.content[:100]}...")
                        if hasattr(last_msg, 'content'):
                            full_response = last_msg.content
            
            status_container.update(label="Complete!", state="complete", expanded=False)
            message_placeholder.markdown(full_response)
            
            # Display generated visualizations
            img_matches = re.findall(r'\((static/.*?\.png)\)', full_response)
            if img_matches:
                for img_path in img_matches:
                    if os.path.exists(img_path):
                        st.image(img_path, caption="Generated Visualization")

            # Display NASA data
            final_data = {}
            if "astronomical_data" in final_state:
                final_data = final_state["astronomical_data"]
            elif "data_team" in final_state and "astronomical_data" in final_state["data_team"]:
                final_data = final_state["data_team"]["astronomical_data"]
            
            if final_data:
                if "photos" in final_data:
                    st.subheader("Mars Rover Photos")
                    photos = final_data["photos"][:3]
                    cols = st.columns(len(photos))
                    for idx, photo in enumerate(photos):
                        with cols[idx]:
                            st.image(photo.get('img_src'), caption=f"{photo.get('camera', {}).get('full_name')} ({photo.get('earth_date')})")
                
                if "url" in final_data and "title" in final_data:
                    st.subheader(f"APOD: {final_data.get('title')}")
                    if final_data.get("media_type") == "image":
                        st.image(final_data.get("url"), caption=final_data.get("explanation"))
                    elif final_data.get("media_type") == "video":
                        st.video(final_data.get("url"))

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            status_container.update(label="Error", state="error")
            st.error(f"Error: {str(e)}")
            st.code(traceback.format_exc())
