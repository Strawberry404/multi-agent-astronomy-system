from typing import Dict, Any
from langchain_core.messages import AIMessage
from state.state_definitions import DataTeamState

try:
    from astropy import units as u
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
        factor = conversions.get((from_unit, to_unit))
        if factor is None:
            # Try reverse
            factor = conversions.get((to_unit, from_unit))
            if factor:
                return value / factor
            return value # Fallback if no conversion found
            
        return value * factor
    
    unit_map = {
        "km": u.km,
        "au": u.au,
        "ly": u.lightyear,
        "pc": u.parsec
    }
    
    if from_unit not in unit_map or to_unit not in unit_map:
        raise ValueError(f"Unit not supported: {from_unit} or {to_unit}")

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
            
            if numbers:
                val = float(numbers[0])
                
                # Simple logic to detect units (can be improved)
                from_unit = "ly" if "light year" in query else "km" if "km" in query or "kilometer" in query else "au" if "au" in query else None
                to_unit = "km" if "to km" in query or "in km" in query or "kilometers" in query else "ly" if "light year" in query and from_unit != "ly" else None
                
                if from_unit and to_unit:
                    res = calculate_distance_conversion(val, from_unit, to_unit)
                    response_text = f"📏 {val} {from_unit} = {res:.2e} {to_unit}"
                    calculations['conversion'] = {
                        'from': f"{val} {from_unit}",
                        'to': f"{res:.2e} {to_unit}"
                    }
                else:
                     response_text = "Could not detect units. Please specify like 'Convert 4.2 light years to km'."
            else:
                response_text = "No number found to convert."
        
        else:
            response_text = "🔢 Calculator ready! I can convert distances (ly, km, au, pc)."
            
    except Exception as e:
        response_text = f"❌ Calculation error: {str(e)}"
    
    return {
        "messages": [AIMessage(content=response_text, name="calculator")],
        "calculations": calculations
    }
