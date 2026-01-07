# 🚀 Astronomy Multi-Agent System - Complete Developer Implementation Guide

**Project:** Multi-Agent Astronomy Assistant with RAG and LangGraph  
**Version:** 1.0  
**Date:** January 2026  
**Status:** Phase 1 Complete (Knowledge Team), Phase 2 In Progress (Data Team)

---

## 📋 Table of Contents

1. [Project Overview](https://claude.ai/chat/2c2a337c-b683-473f-8f0e-e297ea4463ff#project-overview)
2. [Architecture](https://claude.ai/chat/2c2a337c-b683-473f-8f0e-e297ea4463ff#architecture)
3. [Current Status](https://claude.ai/chat/2c2a337c-b683-473f-8f0e-e297ea4463ff#current-status)
4. [Phase 1: Setup & Knowledge Team](https://claude.ai/chat/2c2a337c-b683-473f-8f0e-e297ea4463ff#phase-1-setup--knowledge-team) ✅ COMPLETE
5. [Phase 2: Data & Computation Team](https://claude.ai/chat/2c2a337c-b683-473f-8f0e-e297ea4463ff#phase-2-data--computation-team) 🔄 IN PROGRESS
6. [Phase 3: Output Generation Team](https://claude.ai/chat/2c2a337c-b683-473f-8f0e-e297ea4463ff#phase-3-output-generation-team) 📅 UPCOMING
7. [Phase 4: Main Orchestrator](https://claude.ai/chat/2c2a337c-b683-473f-8f0e-e297ea4463ff#phase-4-main-orchestrator) 📅 UPCOMING
8. [Testing & Validation](https://claude.ai/chat/2c2a337c-b683-473f-8f0e-e297ea4463ff#testing--validation)
9. [Deployment](https://claude.ai/chat/2c2a337c-b683-473f-8f0e-e297ea4463ff#deployment)
10. [Troubleshooting](https://claude.ai/chat/2c2a337c-b683-473f-8f0e-e297ea4463ff#troubleshooting)

---

## 1. Project Overview

### Objective

Build a sophisticated multi-agent astronomy assistant that:

- Answers astronomy questions using a 1000-page PDF knowledge base (RAG)
- Searches the web for recent discoveries
- Queries NASA APIs for real-time astronomical data
- Performs astronomical calculations
- Provides sky visibility information
- Generates educational explanations and observing plans
- Creates visualizations

### Technology Stack

- **Framework:** LangGraph (multi-agent orchestration)
- **LLM:** Google Gemini 2.5 Flash
- **RAG:** FAISS vector store + HuggingFace embeddings
- **APIs:** NASA APIs, Tavily Search, AstroPy, PyEphem

### Architecture Pattern

Three-tier multi-agent system:

1. **Knowledge Team** (Subgroup 1): RAG retrieval, web search, citations
2. **Data Team** (Subgroup 2): NASA APIs, calculations, sky positions
3. **Output Team** (Subgroup 3): Explanations, plans, visualizations

Each subgroup has a supervisor that routes between specialized agents.

---

## 2. Architecture

### System Diagram

```
User Query
    ↓
Main Astronomer Orchestrator
    ↓
    ├─→ Knowledge Team (Subgroup 1)
    │   ├── RAG Retriever Agent
    │   ├── Web Search Agent
    │   └── Citation Manager Agent
    │
    ├─→ Data Team (Subgroup 2)
    │   ├── Database Agent (NASA APIs)
    │   ├── Calculator Agent (AstroPy)
    │   └── Sky Position Agent (PyEphem)
    │
    └─→ Output Team (Subgroup 3)
        ├── Educational Explainer Agent
        ├── Observation Planner Agent
        └── Visualization Agent
    ↓
Comprehensive Response
```

### Directory Structure

```
astronomy-agent/
├── .env                          # API keys (DO NOT COMMIT)
├── .gitignore                    # Git ignore file
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── __init__.py
│
├── config/
│   ├── __init__.py
│   └── config.py                 # Configuration & API key loading
│
├── state/
│   ├── __init__.py
│   └── state_definitions.py     # TypedDict state definitions
│
├── data/
│   └── Astronomy.pdf            # 1000-page knowledge base
│
├── vector_store/
│   └── faiss_index/             # Vector embeddings (generated)
│
├── agents/
│   ├── __init__.py
│   ├── astronomer.py            # Main orchestrator (Phase 4)
│   ├── supervisor.py            # Supervisor utilities
│   │
│   ├── knowledge_team/
│   │   ├── __init__.py
│   │   ├── rag_retriever.py
│   │   ├── web_search.py
│   │   └── citation_manager.py
│   │
│   ├── data_team/
│   │   ├── __init__.py
│   │   ├── database_agent.py
│   │   ├── calculator.py
│   │   └── sky_position.py
│   │
│   └── output_team/
│       ├── __init__.py
│       ├── explainer.py
│       ├── observation_planner.py
│       └── visualizer.py
│
├── graphs/
│   ├── __init__.py
│   ├── knowledge_team_graph.py
│   ├── data_team_graph.py
│   ├── output_team_graph.py
│   └── main_graph.py            # Main orchestrator graph
│
├── utils/
│   ├── __init__.py
│   └── helpers.py               # Shared utilities
│
├── scripts/
│   ├── setup_vector_store.py   # One-time PDF processing
│   └── test_*.py                # Test scripts
│
├── notebooks/
│   └── experiments.ipynb        # Development notebook
│
└── main.py                      # User interface / CLI
```

---

## 3. Current Status

### ✅ Completed

- [x] Project structure setup
- [x] Configuration system with .env
- [x] PDF processing and vector store creation
- [x] Knowledge Team (Subgroup 1)
    - [x] RAG Retriever Agent
    - [x] Web Search Agent (Tavily)
    - [x] Citation Manager Agent
    - [x] Knowledge Team Supervisor
    - [x] Knowledge Team Graph

### 🔄 In Progress

- [ ] Data Team (Subgroup 2)
    - [ ] Database Agent (NASA APIs)
    - [ ] Calculator Agent (AstroPy)
    - [ ] Sky Position Agent (PyEphem)
    - [ ] Data Team Supervisor
    - [ ] Data Team Graph

### 📅 Upcoming

- [ ] Output Team (Subgroup 3)
- [ ] Main Orchestrator (Phase 4)
- [ ] Integration testing
- [ ] Production deployment

---

## 4. Phase 1: Setup & Knowledge Team ✅ COMPLETE

### 4.1 Initial Setup

#### Step 1.1: Environment Setup (15 minutes)

```bash
# Create project directory
mkdir astronomy-agent
cd astronomy-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install base dependencies
pip install langchain langchain-community langchain-google-genai
pip install langgraph faiss-cpu sentence-transformers
pip install pypdf tavily-python python-dotenv
```

#### Step 1.2: Create Directory Structure (5 minutes)

```bash
# Create all directories
mkdir -p config state data vector_store agents/knowledge_team agents/data_team agents/output_team graphs utils scripts notebooks

# Create __init__.py files
touch __init__.py
touch config/__init__.py
touch state/__init__.py
touch agents/__init__.py
touch agents/knowledge_team/__init__.py
touch agents/data_team/__init__.py
touch agents/output_team/__init__.py
touch graphs/__init__.py
touch utils/__init__.py
```

#### Step 1.3: API Keys Setup (10 minutes)

1. **Get Google Gemini API Key:**
    
    - Go to: https://aistudio.google.com/app/apikey
    - Create new API key
    - Copy key
2. **Get Tavily API Key:**
    
    - Go to: https://tavily.com/
    - Sign up for free account
    - Copy API key
3. **Get NASA API Key:**
    
    - Go to: https://api.nasa.gov/
    - Enter email address
    - Instantly receive API key
    - Copy key
4. **Create .env file:**
    

```bash
# .env
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
NASA_API_KEY=your_nasa_api_key_here
```

5. **Create .gitignore:**

```
# .gitignore
.env
__pycache__/
*.py[cod]
.Python
venv/
.ipynb_checkpoints
faiss_index/
.DS_Store
```

#### Step 1.4: Configuration File (10 minutes)

**File:** `config/config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Central configuration"""
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    NASA_API_KEY = os.getenv("NASA_API_KEY")
    
    # Models
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL = "gemini-2.5-flash"
    DEVICE = "cuda"  # or "cpu" if no GPU
    
    # Vector Store
    VECTOR_STORE_PATH = "vector_store/faiss_index"
    RAG_K = 10
    
    # Search
    TAVILY_MAX_RESULTS = 5
    
    # Paths
    PDF_PATH = "data/Astronomy.pdf"
    
    @classmethod
    def validate(cls):
        """Validate API keys"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found")
        if not cls.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY not found")
        if not cls.NASA_API_KEY:
            raise ValueError("NASA_API_KEY not found")
        print("✓ All API keys loaded successfully")

Config.validate()
```

### 4.2 Vector Store Setup

#### Step 2.1: PDF Processing Script (30 minutes)

**File:** `scripts/setup_vector_store.py`

```python
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import Config

def setup_vector_store():
    """One-time setup: Process PDF and create vector store"""
    
    print("📄 Loading PDF...")
    loader = PyPDFLoader(Config.PDF_PATH)
    docs = loader.load()
    print(f"✓ Loaded {len(docs)} pages")
    
    print("\n✂️ Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"✓ Created {len(all_splits)} chunks")
    
    print("\n🔢 Generating embeddings...")
    embed_model = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": Config.DEVICE}
    )
    
    print("This may take several minutes for a 1000-page PDF...")
    vector_store = FAISS.from_documents(
        documents=all_splits,
        embedding=embed_model
    )
    
    print("\n💾 Saving vector store...")
    vector_store.save_local(Config.VECTOR_STORE_PATH)
    print(f"✓ Saved to {Config.VECTOR_STORE_PATH}")
    
    print("\n✨ Setup complete!")
    print(f"Total chunks: {len(all_splits)}")
    print(f"Embedding model: {Config.EMBEDDING_MODEL}")

if __name__ == "__main__":
    setup_vector_store()
```

**Run it:**

```bash
python scripts/setup_vector_store.py
```

**Expected time:** 5-15 minutes depending on hardware

### 4.3 State Definitions (10 minutes)

**File:** `state/state_definitions.py`

```python
from typing import TypedDict, List, Annotated, Optional, Dict, Any
import operator
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

class KnowledgeTeamState(TypedDict):
    """State for Knowledge Team"""
    messages: Annotated[list[BaseMessage], operator.add]
    team_members: List[str]
    next: str
    sources: List[str]
    citations: List[dict]

class RAGState(TypedDict):
    """Internal RAG state"""
    question: str
    context: List[Document]
    answer: str

class DataTeamState(TypedDict):
    """State for Data Team"""
    messages: Annotated[list[BaseMessage], operator.add]
    team_members: List[str]
    next: str
    astronomical_data: Dict[str, Any]
    calculations: Dict[str, Any]
    visibility_info: Dict[str, Any]
    location: Optional[Dict[str, float]]
    date_time: Optional[str]
```

### 4.4 Knowledge Team Implementation

The Knowledge Team is already complete with these files:

- `agents/supervisor.py`
- `agents/knowledge_team/rag_retriever.py`
- `agents/knowledge_team/web_search.py`
- `agents/knowledge_team/citation_manager.py`
- `utils/helpers.py`
- `graphs/knowledge_team_graph.py`
- `main.py`

**Status:** ✅ All files created and tested

---

## 5. Phase 2: Data & Computation Team 🔄 IN PROGRESS

### Overview

The Data Team provides real-time astronomical data, performs calculations, and determines object visibility.

**Estimated Time:** 2-3 days

### 5.1 Dependencies Installation (5 minutes)

```bash
pip install astropy pyephem requests
```

Update `requirements.txt`:

```
# Add these lines
astropy
pyephem
requests
```

### 5.2 Database Agent Implementation (2-3 hours)

**File:** `agents/data_team/database_agent.py`

**Features to implement:**

1. NASA APOD (Astronomy Picture of the Day)
2. Near Earth Objects (Asteroids)
3. Mars Rover Photos ← HIGH PRIORITY
4. Exoplanet Archive ← HIGH PRIORITY
5. Object database (simplified)

**Priority APIs:**

- ✅ APOD (simple, already in code)
- ✅ Near Earth Objects (simple, already in code)
- 🔥 Mars Rover Photos (30 min, very popular)
- 🔥 Exoplanet Archive (1 hour, trending)
- ⏸️ Others (optional)

**Implementation Steps:**

#### Step 1: Basic Structure (30 min)

```python
# agents/data_team/database_agent.py

import requests
from typing import Dict, Any
from langchain_core.messages import AIMessage
from config.config import Config
from state.state_definitions import DataTeamState

def query_nasa_apod() -> Dict[str, Any]:
    """Query NASA APOD"""
    url = f"https://api.nasa.gov/planetary/apod?api_key={Config.NASA_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def query_nasa_neo(date: str = "today") -> Dict[str, Any]:
    """Query Near Earth Objects"""
    from datetime import datetime, timedelta
    
    if date == "today":
        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_date = end_date = date
    
    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={Config.NASA_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def database_agent_node(state: DataTeamState) -> dict:
    """Database Agent main function"""
    print("\n🛰️ [DATABASE AGENT] Querying databases...")
    
    if not state["messages"]:
        return {"messages": []}
    
    query = state["messages"][-1].content.lower()
    data = {}
    response_text = ""
    
    try:
        if "picture of the day" in query or "apod" in query:
            apod_data = query_nasa_apod()
            response_text = format_apod_response(apod_data)
            data['apod'] = apod_data
            
        elif "near earth" in query or "asteroid" in query:
            neo_data = query_nasa_neo()
            response_text = format_neo_response(neo_data)
            data['neo'] = neo_data
            
        # Add more query types here
        
    except Exception as e:
        response_text = f"❌ Error: {str(e)}"
        print(f"[DATABASE AGENT] Error: {e}")
    
    return {
        "messages": [AIMessage(content=response_text, name="database_agent")],
        "astronomical_data": data
    }
```

#### Step 2: Add Mars Rover Photos (30 min)

```python
def query_mars_rover_photos(rover: str = "curiosity", sol: int = 1000):
    """Query Mars Rover Photos API"""
    url = f"https://api.nasa.gov/mars-photos/api/v1/rovers/{rover}/photos"
    params = {
        "sol": sol,
        "api_key": Config.NASA_API_KEY
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

# Add to database_agent_node:
elif "mars" in query and ("photo" in query or "image" in query or "rover" in query):
    rover = "curiosity"
    if "perseverance" in query:
        rover = "perseverance"
    
    mars_data = query_mars_rover_photos(rover=rover)
    photos = mars_data.get('photos', [])
    
    response_text = f"📸 Mars Rover Photos ({rover.capitalize()}):\n"
    response_text += f"Found {len(photos)} photos\n\n"
    
    for i, photo in enumerate(photos[:3]):
        response_text += f"{i+1}. Camera: {photo.get('camera', {}).get('full_name', 'N/A')}\n"
        response_text += f"   Date: {photo.get('earth_date', 'N/A')}\n"
        response_text += f"   Image: {photo.get('img_src', 'N/A')}\n\n"
    
    data['mars_photos'] = mars_data
```

#### Step 3: Add Exoplanet Archive (1 hour)

```python
def query_exoplanet_archive():
    """Query NASA Exoplanet Archive"""
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    query = """
    SELECT pl_name, hostname, sy_dist, pl_rade, pl_masse, disc_year
    FROM ps
    WHERE default_flag = 1
    ORDER BY disc_year DESC
    LIMIT 100
    """
    
    params = {
        "query": query,
        "format": "json"
    }
    
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.json()

# Add to database_agent_node:
elif "exoplanet" in query:
    exo_data = query_exoplanet_archive()
    
    response_text = f"🪐 Exoplanet Database:\n"
    response_text += f"Total confirmed: {len(exo_data)}\n\n"
    response_text += "Recent discoveries:\n"
    
    for i, planet in enumerate(exo_data[:5]):
        response_text += f"{i+1}. {planet.get('pl_name', 'Unknown')}\n"
        response_text += f"   Host: {planet.get('hostname', 'N/A')}\n"
        response_text += f"   Distance: {planet.get('sy_dist', 0):.2f} pc\n"
        response_text += f"   Year: {planet.get('disc_year', 'N/A')}\n\n"
    
    data['exoplanets'] = exo_data
```

#### Step 4: Add Object Database (30 min)

```python
def query_object_info(object_name: str) -> Dict[str, Any]:
    """Query astronomical object database"""
    objects_db = {
        "andromeda": {
            "name": "Andromeda Galaxy (M31)",
            "type": "Spiral Galaxy",
            "distance": "2.537 million light years",
            "magnitude": 3.44,
            "coordinates": {"ra": "00h 42m 44s", "dec": "+41° 16' 9\""}
        },
        "orion nebula": {
            "name": "Orion Nebula (M42)",
            "type": "Emission Nebula",
            "distance": "1,344 light years",
            "magnitude": 4.0
        },
        "betelgeuse": {
            "name": "Betelgeuse",
            "type": "Red Supergiant Star",
            "distance": "548 light years",
            "magnitude": 0.50
        },
        "jupiter": {
            "name": "Jupiter",
            "type": "Gas Giant Planet",
            "magnitude": -2.94,
            "moons": 95
        },
        "mars": {
            "name": "Mars",
            "type": "Terrestrial Planet",
            "magnitude": -2.94,
            "moons": 2
        }
    }
    
    object_lower = object_name.lower()
    for key, data in objects_db.items():
        if key in object_lower or object_lower in key:
            return data
    
    return {"error": f"Object '{object_name}' not found"}
```

**Deliverable:** Complete `agents/data_team/database_agent.py` file

### 5.3 Calculator Agent Implementation (2-3 hours)

**File:** `agents/data_team/calculator.py`

**Features:**

1. Distance conversions (AU, light years, km, parsecs)
2. Light travel time calculations
3. Angular separation
4. Coordinate transformations
5. Magnitude calculations

**Implementation Steps:**

#### Step 1: Basic Calculator Structure (1 hour)

```python
# agents/data_team/calculator.py

from typing import Dict, Any
from langchain_core.messages import AIMessage
from state.state_definitions import DataTeamState

try:
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

def calculate_distance_conversion(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between distance units"""
    if not ASTROPY_AVAILABLE:
        # Fallback conversions
        conversions = {
            ("ly", "km"): 9.461e12,
            ("au", "km"): 1.496e8,
            ("pc", "ly"): 3.26
        }
        factor = conversions.get((from_unit, to_unit), 1.0)
        return value * factor
    
    unit_map = {
        "km": u.km,
        "au": u.au,
        "ly": u.lightyear,
        "pc": u.parsec
    }
    
    distance = value * unit_map[from_unit]
    return distance.to(unit_map[to_unit]).value

def calculator_node(state: DataTeamState) -> dict:
    """Calculator Agent main function"""
    print("\n🔢 [CALCULATOR] Performing calculations...")
    
    if not state["messages"]:
        return {"messages": []}
    
    query = state["messages"][-1].content.lower()
    calculations = {}
    response_text = ""
    
    try:
        if "convert" in query:
            # Parse conversion request
            import re
            numbers = re.findall(r'\d+\.?\d*', query)
            
            if numbers and "light year" in query and "km" in query:
                ly_value = float(numbers[0])
                km_value = calculate_distance_conversion(ly_value, "ly", "km")
                
                response_text = f"📏 {ly_value} light years = {km_value:.2e} km"
                calculations['conversion'] = {
                    'from': f"{ly_value} ly",
                    'to': f"{km_value:.2e} km"
                }
        
        else:
            response_text = "🔢 Calculator ready! I can convert distances and perform calculations."
            
    except Exception as e:
        response_text = f"❌ Calculation error: {str(e)}"
    
    return {
        "messages": [AIMessage(content=response_text, name="calculator")],
        "calculations": calculations
    }
```

#### Step 2: Add More Calculation Types (1-2 hours)

Add functions for:

- Light travel time
- Angular separation
- Magnitude calculations
- Coordinate conversions

**Deliverable:** Complete `agents/data_team/calculator.py` file

### 5.4 Sky Position Agent Implementation (3-4 hours)

**File:** `agents/data_team/sky_position.py`

**Features:**

1. Real-time object positions (altitude, azimuth)
2. Rise/set times
3. Visibility calculations
4. Coordinate systems (RA/Dec, Alt/Az)

**Implementation Steps:**

#### Step 1: Basic Sky Position (2 hours)

```python
# agents/data_team/sky_position.py

from datetime import datetime
from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage
from state.state_definitions import DataTeamState

try:
    import ephem
    EPHEM_AVAILABLE = True
except ImportError:
    EPHEM_AVAILABLE = False

def get_object_position(object_name: str, lat: float, lon: float, 
                       date_time: Optional[datetime] = None) -> Dict:
    """Calculate object position and visibility"""
    if not EPHEM_AVAILABLE:
        return {"error": "PyEphem not installed"}
    
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.date = date_time or datetime.utcnow()
    
    object_map = {
        "sun": ephem.Sun(),
        "moon": ephem.Moon(),
        "jupiter": ephem.Jupiter(),
        "mars": ephem.Mars(),
        "venus": ephem.Venus(),
        "saturn": ephem.Saturn()
    }
    
    obj = object_map.get(object_name.lower())
    if not obj:
        return {"error": f"Object '{object_name}' not supported"}
    
    obj.compute(observer)
    
    try:
        next_rising = observer.next_rising(obj).datetime()
        next_setting = observer.next_setting(obj).datetime()
    except:
        next_rising = next_setting = None
    
    return {
        "name": object_name,
        "altitude": float(obj.alt) * 180 / 3.14159,
        "azimuth": float(obj.az) * 180 / 3.14159,
        "is_visible": float(obj.alt) > 0,
        "next_rising": next_rising,
        "next_setting": next_setting
    }

def sky_position_node(state: DataTeamState) -> dict:
    """Sky Position Agent main function"""
    print("\n🌍 [SKY POSITION] Calculating positions...")
    
    if not state["messages"]:
        return {"messages": []}
    
    query = state["messages"][-1].content.lower()
    location = state.get("location", {"lat": 40.7128, "lon": -74.0060})
    
    visibility_info = {}
    response_text = ""
    
    try:
        if not EPHEM_AVAILABLE:
            response_text = "⚠️ Install PyEphem: pip install pyephem"
        else:
            # Determine which objects to check
            objects = []
            if "jupiter" in query:
                objects.append("jupiter")
            if "mars" in query:
                objects.append("mars")
            # Add more...
            
            if not objects:
                objects = ["moon", "jupiter", "mars", "venus"]
            
            response_text = f"🌍 Sky Positions:\n\n"
            for obj in objects:
                pos = get_object_position(obj, location["lat"], location["lon"])
                if "error" not in pos:
                    visibility_info[obj] = pos
                    response_text += f"📍 {obj.capitalize()}:\n"
                    response_text += f"  Altitude: {pos['altitude']:.1f}°\n"
                    response_text += f"  Visible: {'Yes' if pos['is_visible'] else 'No'}\n\n"
    
    except Exception as e:
        response_text = f"❌ Error: {str(e)}"
    
    return {
        "messages": [AIMessage(content=response_text, name="sky_position")],
        "visibility_info": visibility_info
    }
```

#### Step 2: Enhanced Features (1-2 hours)

Add:

- Multiple location support
- Time-based queries ("tonight at 9pm")
- Best viewing times
- Circumpolar objects handling

**Deliverable:** Complete `agents/data_team/sky_position.py` file

### 5.5 Data Team Graph (1 hour)

**File:** `graphs/data_team_graph.py`

```python
# graphs/data_team_graph.py

import functools
from langgraph.graph import START, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI

from config.config import Config
from state.state_definitions import DataTeamState
from agents.data_team import database_agent_node, calculator_node, sky_position_node
from agents.supervisor import create_team_supervisor

def data_supervisor_node(state: DataTeamState, llm) -> dict:
    """Data Team Supervisor"""
    supervisor_chain = create_team_supervisor(
        llm,
        (
            "You are a data team supervisor. Route queries:\n"
            "- 'database_agent': For NASA data, object info\n"
            "- 'calculator': For calculations, conversions\n"
            "- 'sky_position': For visibility, positions\n"
            "Respond FINISH when done."
        ),
        state["team_members"]
    )
    
    result = supervisor_chain.invoke(state)
    next_agent = result.get("next", "FINISH")
    return {"next": next_agent}

def build_data_team():
    """Build Data Team
```


<parameter name="command">update</parameter> <parameter name="id">developer_implementation_guide</parameter> <parameter name="old_str">def build_data_team(): """Build Data Team</parameter> <parameter name="new_str">def build_data_team(): """Build Data Team graph""" llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL)

```
workflow = StateGraph(DataTeamState)

workflow.add_node("supervisor", functools.partial(data_supervisor_node, llm=llm))
workflow.add_node("database_agent", database_agent_node)
workflow.add_node("calculator", calculator_node)
workflow.add_node("sky_position", sky_position_node)

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "database_agent": "database_agent",
        "calculator": "calculator",
        "sky_position": "sky_position",
        "FINISH": END
    }
)

workflow.add_edge("database_agent", "supervisor")
workflow.add_edge("calculator", "supervisor")
workflow.add_edge("sky_position", "supervisor")
workflow.add_edge(START, "supervisor")

return workflow.compile()
```

data_team = build_data_team()

````

**Deliverable:** Complete `graphs/data_team_graph.py` file

### 5.6 Testing Data Team (30 minutes)

**File:** `scripts/test_data_team.py`

```python
# scripts/test_data_team.py

from langchain_core.messages import HumanMessage
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
    
    for event in data_team.stream(initial_state, stream_mode='values'):
        if event.get("messages"):
            latest = event["messages"][-1]
            print(f"\nAgent: {latest.name}")
            print(f"Output: {latest.content}\n")

if __name__ == "__main__":
    # Test cases
    test_data_team("Tell me about Jupiter")
    test_data_team("Convert 4.2 light years to kilometers")
    test_data_team("When does Mars rise tonight?")
    test_data_team("Show me Mars rover photos")
    test_data_team("How many exoplanets have we found?")
````

**Run tests:**

```bash
python scripts/test_data_team.py
```

**Expected Results:**

- Database queries return NASA data
- Calculations convert units correctly
- Sky positions show visibility info

---

## 6. Phase 3: Output Generation Team 📅 UPCOMING

**Estimated Time:** 2-3 days

### Overview

The Output Team formats and presents information in user-friendly ways.

### Components

1. **Educational Explainer Agent** - Clear explanations
2. **Observation Planner Agent** - Observing schedules
3. **Visualization Agent** - Charts and diagrams

### Dependencies

```bash
pip install matplotlib plotly pandas
```

### File Structure

- `agents/output_team/explainer.py`
- `agents/output_team/observation_planner.py`
- `agents/output_team/visualizer.py`
- `graphs/output_team_graph.py`

**Status:** Not yet started  
**Start After:** Data Team completion

---

## 7. Phase 4: Main Orchestrator 📅 UPCOMING

**Estimated Time:** 1-2 days

### Overview

Top-level orchestrator that routes between all three subgroups.

### Components

1. **Main Astronomer Agent** - Query analysis and routing
2. **Main Graph** - Connects all subgroups
3. **Integration Logic** - Combines results

### File Structure

- `agents/astronomer.py`
- `graphs/main_graph.py`
- Update `main.py`

**Status:** Not yet started  
**Start After:** Output Team completion

---

## 8. Testing & Validation

### Unit Tests

- Test each agent independently
- Test each subgroup graph
- Test main orchestrator

### Integration Tests

- End-to-end query flow
- Multi-subgroup queries
- Error handling

### Test Queries

```python
test_queries = [
    # Knowledge only
    "What is a black hole?",
    
    # Data only
    "When does Jupiter rise tonight?",
    
    # Knowledge + Data
    "Tell me about Mars and show me rover photos",
    
    # All subgroups
    "Explain neutron stars and create an observing plan",
    
    # Complex
    "Compare exoplanets to Earth, show data, and visualize"
]
```

---

## 9. Deployment

### Local Deployment

```bash
# Run the system
python main.py

# Or import and use
from main import ask_astronomy_question
result = ask_astronomy_question("What is a neutron star?")
```

### API Deployment (Future)

- FastAPI wrapper
- Docker containerization
- Cloud deployment (AWS/GCP/Azure)

---

## 10. Troubleshooting

### Common Issues

**Issue 1: Import Errors**

```
ModuleNotFoundError: No module named 'config'
```

**Solution:**

- Ensure all `__init__.py` files exist
- Run from project root directory
- Check Python path

**Issue 2: API Key Errors**

```
ValueError: GOOGLE_API_KEY not found
```

**Solution:**

- Check `.env` file exists
- Verify key names match exactly
- No quotes around keys in `.env`

**Issue 3: Vector Store Not Found**

```
FileNotFoundError: faiss_index not found
```

**Solution:**

- Run `python scripts/setup_vector_store.py`
- Check `Config.VECTOR_STORE_PATH`
- Ensure PDF is in `data/` folder

**Issue 4: CUDA/GPU Errors**

```
RuntimeError: CUDA out of memory
```

**Solution:**

- Change `Config.DEVICE = "cpu"` in config.py
- Or reduce batch size

---

## 11. Timeline & Milestones

### Week 1

- [x] Day 1-2: Setup & Knowledge Team
- [ ] Day 3-4: Data Team implementation
- [ ] Day 5: Data Team testing

### Week 2

- [ ] Day 1-2: Output Team
- [ ] Day 3-4: Main Orchestrator
- [ ] Day 5: Integration testing

### Week 3

- [ ] Day 1-2: Bug fixes & optimization
- [ ] Day 3-4: Documentation
- [ ] Day 5: Deployment

---

## 12. Success Criteria

### Phase 1 ✅

- [x] RAG retrieves relevant PDF content
- [x] Web search finds recent info
- [x] Citations properly formatted
- [x] Knowledge Team graph works

### Phase 2 🔄

- [ ] NASA APIs return data
- [ ] Calculations are accurate
- [ ] Sky positions are correct
- [ ] Data Team graph works

### Phase 3

- [ ] Explanations are clear
- [ ] Observation plans are practical
- [ ] Visualizations render correctly

### Phase 4

- [ ] Main orchestrator routes correctly
- [ ] All subgroups integrate
- [ ] End-to-end queries work
- [ ] Performance is acceptable

---

## 13. Next Actions for Developer

### Immediate (Today)

1. ✅ Review this guide
2. ✅ Confirm all Phase 1 files are complete
3. 🔄 Start Phase 2: Data Team

### This Week

1. Implement Database Agent (2-3 hours)
    - Focus on Mars Rover & Exoplanets first
2. Implement Calculator Agent (2-3 hours)
3. Implement Sky Position Agent (3-4 hours)
4. Build Data Team Graph (1 hour)
5. Test Data Team (30 min)

### Blockers to Report

- Missing API keys
- PDF not available
- Hardware limitations (GPU/RAM)
- Library installation issues

---

## 14. Resources

### Documentation

- LangGraph: https://langchain-ai.github.io/langgraph/
- NASA APIs: https://api.nasa.gov/
- AstroPy: https://docs.astropy.org/
- PyEphem: https://rhodesmill.org/pyephem/

### Support

- Project issues: [Create issue tracker]
- Questions: [Add communication channel]

---

## 15. Appendix

### A. Complete File Checklist

#### Configuration

- [x] `.env`
- [x] `.gitignore`
- [x] `config/config.py`
- [x] `state/state_definitions.py`

#### Knowledge Team (Phase 1) ✅

- [x] `agents/supervisor.py`
- [x] `agents/knowledge_team/rag_retriever.py`
- [x] `agents/knowledge_team/web_search.py`
- [x] `agents/knowledge_team/citation_manager.py`
- [x] `graphs/knowledge_team_graph.py`
- [x] `utils/helpers.py`

#### Data Team (Phase 2) 🔄

- [ ] `agents/data_team/database_agent.py`
- [ ] `agents/data_team/calculator.py`
- [ ] `agents/data_team/sky_position.py`
- [ ] `graphs/data_team_graph.py`

#### Output Team (Phase 3) 📅

- [ ] `agents/output_team/explainer.py`
- [ ] `agents/output_team/observation_planner.py`
- [ ] `agents/output_team/visualizer.py`
- [ ] `graphs/output_team_graph.py`

#### Main (Phase 4) 📅

- [ ] `agents/astronomer.py`
- [ ] `graphs/main_graph.py`
- [x] `main.py`

#### Scripts

- [x] `scripts/setup_vector_store.py`
- [ ] `scripts/test_data_team.py`
- [ ] `scripts/test_output_team.py`
- [ ] `scripts/test_integration.py`

### B. Dependencies

```
langchain>=0.1.0
langchain-community>=0.1.0
langchain-google-genai>=0.1.0
langgraph>=0.1.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
pypdf>=3.0.0
tavily-python>=0.1.0
python-dotenv>=1.0.0
astropy>=5.0.0
pyephem>=4.1.0
requests>=2.31.0
matplotlib>=3.7.0
plotly>=5.14.0
pandas>=2.0.0
```

---

**END OF DEVELOPER GUIDE**

_Last Updated: January 2026_  
_Version: 1.0_  
_Status: Phase 1 Complete, Phase 2 In Progress_</parameter>


