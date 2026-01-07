import requests
from typing import Dict, Any
from langchain.tools import Tool , tool
# from langchain_core.messages import AIMessage
import sys
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config import Config
from state.state_definitions import DataTeamState
START_DATE = "2024-10-01"
END_DATE = "2024-10-07"

@tool
def query_nasa_apod(date:str) -> str:
    """
    this tool gives the image of a specific date 
    
    :param date: a date in this sturcture YYYY-MM-DD
    :type date: str
    :return: Description
    :rtype: str
    
    """
    # url=f"https://api.nasa.gov/neo/rest/v1/feed?start_date={START_DATE}&end_date={END_DATE}&api_key={Config.NASA_API_KEY}"
    url = f"https://api.nasa.gov/planetary/apod?api_key={Config.NASA_API_KEY}&date={date}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

@tool
def query_nasa_apod2(date1:str , date2:str) -> str:
    """
    this tool gives the images in range between date1 and date2 
    
    :param date1 , date2: a date in this sturcture YYYY-MM-DD
    :type date1 , date2 : str
    :return: Description
    :rtype: str
    if there's no start date the start date value shall be "none"
    if there's no end_date you shall write on date2 on the end date "today"

    
    """
    # url=f"https://api.nasa.gov/neo/rest/v1/feed?start_date={START_DATE}&end_date={END_DATE}&api_key={Config.NASA_API_KEY}"
    url = f"https://api.nasa.gov/planetary/apod?api_key={Config.NASA_API_KEY}&start_date={date1}&end_date={date2}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

@tool
def query_nasa_apod2(date1:str , date2:str) -> str:
    """
    this tool gives the images in range between date1 and date2 
    
    :param date1 , date2: a date in this sturcture YYYY-MM-DD
    :type date1 , date2 : str
    :return: Description
    :rtype: str
    if there's no start date the start date value shall be "none"
    if there's no end_date you shall write on date2 on the end date "today"

    
    """
    # url=f"https://api.nasa.gov/neo/rest/v1/feed?start_date={START_DATE}&end_date={END_DATE}&api_key={Config.NASA_API_KEY}"
    url = f"https://api.nasa.gov/planetary/apod?api_key={Config.NASA_API_KEY}&start_date={date1}&end_date={date2}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


# tools = [
#     Tool(
#         name="NASA_Astronomy_Picture_of_the_Day",
#         func=query_nasa_apod,
#         description="NASA Astronomy Picture of the Day via the format of YYYY-MM-DD , it can be of today , yesterday tomorrow or anyother day"
#     ),
#     Tool(
#         name="NASA_Astronomy_Pictures_of_a_interval_of_days",
#         func=query_nasa_apod2,
#         description="NASA Astronomy Picture of a range of days  via the format of YYYY-MM-DD , date1 is the startdate and date2 is the end date"
#     )
# ]

llm = ChatGoogleGenerativeAI(model = Config.LLM_MODEL)

agent = initialize_agent(
    tools=[query_nasa_apod , query_nasa_apod2],
    llm = llm, 
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# the database agent needs to have different yea know different 

agent.invoke("gimme pictures between first january 2025 and end of same january")


