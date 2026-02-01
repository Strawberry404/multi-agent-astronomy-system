#  specific Astronomy Multi-Agent System

A powerful multi-agent AI system designed to answer astronomy questions, fetch real-time NASA data, and generate educational visualizations. Powered by **LangGraph**, **Google Gemini**, and **NASA APIs**.

##  Features

The system employs a team of specialized AI agents:

1.  ** Knowledge Team**:
    *   Retrieves information from a curated PDF knowledge base.
    *   Performs web searches for the latest astronomical news (via Tavily).
    *   Good for: "What is a black hole?", "Who discovered Neptune?"

2.  ** Data Team**:
    *   Connects to **NASA APIs** (daily imagery, Mars rovers, asteroids, exoplanets).
    *   Calculates real-time sky positions of planets.
    *   Good for: "Show me Mars photos", "Is Jupiter visible tonight?", "Distance to Andromeda".

3.  ** Output Team**:
    *   Synthesizes information into clear, educational explanations.
    *   Creates custom charts and visualizations (e.g., Solar System orbits, star charts).
    *   Plans observation sessions based on your location.

## Prerequisites

*   **Python 3.10+**
*   **Git**
*   API Keys:
    *   **Google Gemini API** (LLM)
    *   **NASA API** (Data)
    *   **Tavily API** (Web Search)
    *   **LangSmith** (Optional, for tracing)

##  Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/strawberry404/multi-agent-astronomy-system.git
    cd multi-agent-astronomy-system
    ```

2.  **Create a virtual environment:**
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # Mac/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  Create a `.env` file in the root directory (copy from `.envexemple`):
    ```bash
    cp .envexemple .env
    ```

2.  Fill in your API keys in `.env`:
    ```env
    GOOGLE_API_KEY=your_gemini_key_here
    NASA_API_KEY=your_nasa_key_here
    TAVILY_API_KEY=your_tavily_key_here
    
    # Optional: LangSmith for tracing
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=your_langchain_key_here
    ```

## Running the App

Start the Streamlit interface:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

##  Testing

To verify the system modules:

```bash
# Run integration tests
python scripts/test_integration.py
```

##  Acknowledgments

*   **NASA GIBS**: Imagery provided by services from NASA's Global Imagery Browse Services (GIBS), part of NASA's Earth Science Data and Information System (ESDIS).
*   **LangChain & LangGraph**: Frameworks for building the agentic workflow.