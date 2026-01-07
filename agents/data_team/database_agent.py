import requests
from typing import Dict, Any, List
from datetime import datetime, timedelta
from langchain_core.messages import AIMessage, HumanMessage
from config.config import Config
from state.state_definitions import DataTeamState

def format_apod_response(data: Dict[str, Any]) -> str:
    """Format APOD response"""
    title = data.get("title", "Astronomy Picture of the Day")
    date = data.get("date", "Unknown Date")
    explanation = data.get("explanation", "No explanation available.")
    url = data.get("url", "")
    
    return f"🌌 **{title}** ({date})\n\n{explanation}\n\nImage URL: {url}"

def format_neo_response(data: Dict[str, Any]) -> str:
    """Format NEO response"""
    element_count = data.get("element_count", 0)
    near_earth_objects = data.get("near_earth_objects", {})
    
    response = f"☄️ **Near Earth Objects Report**\nFound {element_count} objects recently.\n\n"
    
    # Get first few objects
    count = 0
    for date, objects in near_earth_objects.items():
        if count >= 3: break
        for obj in objects:
            if count >= 3: break
            name = obj.get("name", "Unknown")
            diameter = obj.get("estimated_diameter", {}).get("kilometers", {}).get("estimated_diameter_max", 0)
            hazardous = "⚠️ Hazardous" if obj.get("is_potentially_hazardous_asteroid") else "✅ Safe"
            response += f"- **{name}**: {diameter:.2f} km diameter, {hazardous}\n"
            count += 1
            
    return response

def query_nasa_apod() -> Dict[str, Any]:
    """Query NASA APOD"""
    url = f"https://api.nasa.gov/planetary/apod?api_key={Config.NASA_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def query_nasa_neo(date: str = "today") -> Dict[str, Any]:
    """Query Near Earth Objects"""
    if date == "today":
        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_date = end_date = date
    
    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={Config.NASA_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def query_mars_rover_photos(rover: str = "curiosity", sol: int = 1000) -> Dict[str, Any]:
    """Query Mars Rover Photos API"""
    url = f"https://api.nasa.gov/mars-photos/api/v1/rovers/{rover}/photos"
    params = {
        "sol": sol,
        "api_key": Config.NASA_API_KEY
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def query_exoplanet_archive() -> List[Dict[str, Any]]:
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

def database_agent_node(state: DataTeamState) -> dict:
    """Database Agent main function"""
    print("\n🛰️ [DATABASE AGENT] Querying databases...")
    
    messages = state.get("messages", [])
    query = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
             query = m.content.lower()
             break

    if not query:
        return {"messages": []}
        
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
            data['photos'] = photos  # Also store photos directly for easier access
            
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
        
        else:
            # Fallback to object info if a known object is mentioned
            objects = ["andromeda", "orion nebula", "betelgeuse", "jupiter", "mars"]
            found = False
            for obj in objects:
                if obj in query:
                    obj_data = query_object_info(obj)
                    if "error" not in obj_data:
                        response_text = f"ℹ️ **{obj_data.get('name')}**\n"
                        response_text += f"Type: {obj_data.get('type')}\n"
                        response_text += f"Magnitude: {obj_data.get('magnitude')}\n"
                        if 'distance' in obj_data:
                            response_text += f"Distance: {obj_data.get('distance')}\n"
                        data['object_info'] = obj_data
                        found = True
                        break
            
            if not found:
                response_text = "I can access NASA APOD, Asteroids, Mars Rover photos, Exoplanets, and basic object info. What would you like to know?"

    except Exception as e:
        response_text = f"❌ Error: {str(e)}"
        print(f"[DATABASE AGENT] Error: {e}")
    
    return {
        "messages": [AIMessage(content=response_text, name="database_agent")],
        "astronomical_data": data
    }
