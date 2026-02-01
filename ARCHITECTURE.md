# 🏗️ Architecture: Multi-Agent Astronomy System

This document outlines the technical architecture of the Astronomy Multi-Agent System. The system is built using **LangGraph** for orchestration, **Google Gemini** as the reasoning engine, and **Streamlit** for the frontend.

## 🌟 System Overview

The system follows a **Hierarchical Agent Architecture**. A top-level "Astronomer" agent routes user queries to specialized sub-teams, each responsible for a specific domain.

```mermaid
graph TD
    User([User]) <--> UI[Streamlit UI]
    UI <--> Main[Main Graph / Astronomer Agent]
    
    subgraph "Specialized Teams"
        Main --> KT[Knowledge Team]
        Main --> DT[Data Team]
        Main --> OT[Output Team]
    end
    
    KT --> Main
    DT --> Main
    OT --> Main
```

## 🧩 Graph Workflows

### 1. Main Graph (`graphs/main_graph.py`)
The entry point for all queries. It assesses the user's intent and routes to the appropriate team.

```mermaid
stateDiagram-v2
    [*] --> Astronomer
    
    state "Astronomer Agent" as Astronomer
    state "Knowledge Team" as XT
    state "Data Team" as DT
    state "Output Team" as OT
    
    Astronomer --> XT: Definition / Fact / History
    Astronomer --> DT: Photo / Coordinates / Data
    Astronomer --> OT: Explanation / Plot / Plan
    
    XT --> Astronomer: Return Content
    DT --> Astronomer: Return Content
    OT --> Astronomer: Return Content
    
    Astronomer --> [*]: FINISH
```

### 2. Knowledge Team (`graphs/knowledge_team_graph.py`)
Handles retrieval from unstructured sources (PDFs, Web).

```mermaid
graph LR
    Supervisor((Supervisor))
    
    subgraph "Agents"
        RAG[RAG Retriever]
        Web[Web Search]
        Cite[Citation Manager]
    end
    
    Supervisor --> RAG
    Supervisor --> Web
    RAG --> Supervisor
    Web --> Supervisor
    
    Supervisor --> Cite
    Cite --> Supervisor
```

- **RAG Retriever**: Queries the local vector database (ChromaDB) created from PDFs.
- **Web Search**: Uses Tavily API for real-time information.
- **Citation Manager**: Formats sources.

### 3. Data Team (`graphs/data_team_graph.py`)
Handles structured data fetching and calculations.

```mermaid
graph LR
    Supervisor((Supervisor))
    
    subgraph "Agents"
        DB[Database Agent]
        Calc[Calculator]
        Sky[Sky Position]
    end
    
    Supervisor --> DB
    Supervisor --> Calc
    Supervisor --> Sky
    
    DB --> Supervisor
    Calc --> Supervisor
    Sky --> Supervisor
```

- **Database Agent**: Fetches media and data from NASA APIs (Image Library, Mars Rover, Exoplanets).
- **Calculator**: Performs astrophysical calculations.
- **Sky Position**: Uses `ephem` or formulas to determine object visibility.

### 4. Output Team (`graphs/output_team_graph.py`)
Responsible for final presentation and visualization.

```mermaid
graph LR
    Supervisor((Supervisor))
    
    subgraph "Agents"
        Explain[Explainer]
        Plan[Observation Planner]
        Viz[Visualizer]
    end
    
    Supervisor --> Explain
    Supervisor --> Plan
    Supervisor --> Viz
    
    Explain --> Supervisor
    Plan --> Supervisor
    Viz --> Supervisor
```

- **Explainer**: Generates text-based educational content.
- **Observation Planner**: Guides users on how/when to observe targets.
- **Visualizer**: Generates matplotlib/plotly charts and saves them as images for the UI to render.


## 💾 State Management
The system uses `TypedDict` schemas to pass state between agents.

- **Messages**: List of `BaseMessage` objects (history).
- **Astronomical Data**: Shared dictionary for raw data retrieved by the Data Team.
- **Location**: User's latitude/longitude for observation planning.
