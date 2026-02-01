from datetime import datetime, timezone
from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage
from state.state_definitions import DataTeamState

try:
    import ephem
    EPHEM_AVAILABLE = True
except ImportError:
    EPHEM_AVAILABLE = False

def get_object_position(object_name: str, lat: float, lon: float, 
                       date_time: Optional[datetime] = None) -> Dict[str, Any]:
    """Calculate object position and visibility"""
    if not EPHEM_AVAILABLE:
        return {"error": "PyEphem not installed"}
    
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.date = date_time or datetime.now(timezone.utc)
    
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
    except Exception:
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
    
    messages = state.get("messages", [])
    query = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
             query = m.content.lower()
             break

    if not query:
        return {"messages": []}
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
                    response_text += f"  Visible: {'Yes' if pos['is_visible'] else 'No'}\n"
                    
                    # --- ADD THESE LINES ---
                    if pos.get('next_rising'):
                        # Format the datetime nicely
                        rise_time = pos['next_rising'].strftime('%H:%M UTC')
                        response_text += f"  Rises: {rise_time}\n"
                    # -----------------------
                    
                    response_text += "\n"
    except Exception as e:
        response_text = f"❌ Error: {str(e)}"
    
    return {
        "messages": [AIMessage(content=response_text, name="sky_position")],
        "visibility_info": visibility_info
    }
