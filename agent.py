from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import random
from datetime import datetime, timedelta


@tool
def check_flight_status(flight_number: str) -> str:
    """Check real-time status of a flight.
    Use when customer asks about flight status, delay, or gate.
    Args:
        flight_number: e.g. AA123 or UA456
    """
    statuses = ["On Time", "Delayed 45 minutes", "Boarding", "Departed", "Landed"]
    gates = ["B12", "C34", "A5", "D22", "E7"]
    dep_time = (datetime.now() + timedelta(hours=2)).strftime("%I:%M %p")
    return (
        f"Flight {flight_number.upper()}: {random.choice(statuses)}, "
        f"Gate {random.choice(gates)}, Terminal B, Departure {dep_time}. "
        f"Check the airline's official website for live updates."
    )


@tool
def calculate_baggage_fee(bag_count: int, weight_lbs: float, fare_class: str = "economy") -> str:
    """Calculate baggage fee for United Airlines.
    Use when customer asks how much bags will cost.
    Args:
        bag_count: number of bags
        weight_lbs: weight per bag
        fare_class: basic_economy, economy, business, or first
    """
    fees = {"basic_economy": [35, 45], "economy": [35, 45], "business": [0, 0], "first": [0, 0]}
    fc = fare_class.lower().replace(" ", "_")
    if fc not in fees:
        fc = "economy"
    base = fees[fc]
    total = 0
    lines = []
    for i in range(min(bag_count, 2)):
        ow = 200 if weight_lbs > 70 else 100 if weight_lbs > 50 else 0
        t = (base[i] if i < len(base) else 150) + ow
        total += t
        lines.append(f"Bag {i+1}: ${t}")
    return f"Baggage fees ({fc}): {', '.join(lines)}. Total: ${total}. Premier members may get free bags."


@tool
def get_airport_info(airport_code: str) -> str:
    """Get United Airlines airport info.
    Use when customer asks about airport terminals or hubs.
    Args:
        airport_code: 3-letter code e.g. ORD
    """
    db = {
    "ORD": "Chicago O'Hare (ORD): Major hub, ATS train connects terminals. Allow extra connection time.",
    "EWR": "Newark Liberty (EWR): Terminal A/B/C. AirTrain connects to NJ Transit for NYC.",
    "IAH": "Houston Bush (IAH): Terminals A-E. Inter-terminal train runs frequently.",
    "LAX": "LA International (LAX): 9 terminals in horseshoe layout. Free shuttle connects all. Arrive early.",
    "SFO": "San Francisco (SFO): 4 terminals, AirTrain loops all day. International Terminal is separate.",
    "JFK": "New York JFK (JFK): 6 terminals. AirTrain connects terminals and subway. Arrive 3hrs early for intl.",
    "ATL": "Atlanta (ATL): World's busiest airport. Underground train connects all concourses.",
    "DFW": "Dallas Fort Worth (DFW): 5 terminals. Skylink train runs 24/7. Allow 30+ min for connections.",
}
    return db.get(airport_code.upper(), f"No info available for {airport_code.upper()}. Check the airport's official website.")


def build_agent():
    tools = [check_flight_status, calculate_baggage_fee, get_airport_info]
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    system = (
    "You are a helpful airline customer support AI assistant. "
    "Use tools for flight status, baggage fees, and airport info. "
    "For general policy questions answer from your knowledge. "
    "Be friendly and concise. If unsure, advise the customer to contact the airline directly."
)
    return create_react_agent(llm, tools, prompt=system)