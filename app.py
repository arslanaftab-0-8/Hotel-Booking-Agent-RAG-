import os
import datetime
import random
from typing import List, Dict

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor

# --- Load API key from .env file ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Mock Hotel Database ---
mock_hotel_db = {
    "hotel_001": {
        "name": "The Royal Londoner", "rating": 4.8, "price_per_night": 350,
        "description": "A 5-star hotel offering luxury in the heart of London. Features a rooftop bar.",
        "amenities": ["Free Wi-Fi", "Swimming Pool", "Gym", "Rooftop Bar"],
        "rooms": {"Deluxe King Room": {}, "Executive Suite": {}}
    },
    "hotel_002": {
        "name": "The Thames View Inn", "rating": 4.3, "price_per_night": 180,
        "description": "A charming inn with stunning views of the River Thames.",
        "amenities": ["Free Wi-Fi", "Restaurant", "Pet-Friendly"],
        "rooms": {"Standard Double": {}, "River View Room": {}}
    },
}

# --- Tool Functions ---
def retrieve_hotel_info(query: str) -> str:
    print(f"--- RAG Tool Called with query: {query} ---")
    results = []
    for hotel_id, details in mock_hotel_db.items():
        search_space = f"{details['name']} {details['description']} {' '.join(details['amenities'])}".lower()
        if query.lower() in search_space:
            results.append(f"Name: {details['name']}, Rating: {details['rating']}, Price: £{details['price_per_night']}.")
    return "\n".join(results) if results else "No specific hotel information found for that query."

def search_hotels(destination: str, check_in_date: str, check_out_date: str, num_adults: int) -> List[Dict]:
    print(f"--- Search Tool Called for {destination} ---")
    if "london" in destination.lower():
        return [{"hotel_id": hotel_id, "name": details["name"], "rating": details["rating"]} for hotel_id, details in mock_hotel_db.items()]
    return []

def book_hotel_room(hotel_id: str, room_type: str, check_in_date: str, check_out_date: str, guest_name: str, guest_email: str) -> str:
    print(f"--- Booking Tool Called for hotel: {hotel_id} ---")
    hotel = mock_hotel_db.get(hotel_id)
    if not hotel or room_type not in hotel["rooms"]:
        return "Error: Invalid hotel or room type specified."
    booking_id = f"BK-{random.randint(10000, 99999)}"
    return f"Booking confirmed! Your booking ID is {booking_id}. A confirmation has been sent to {guest_email}."

# --- Tool Wrapping for LangChain ---
tools = [
    Tool.from_function(retrieve_hotel_info, name="retrieve_hotel_info", description="Get hotel info by keyword"),
    Tool.from_function(search_hotels, name="search_hotels", description="Search hotels by destination"),
    Tool.from_function(book_hotel_room, name="book_hotel_room", description="Book a hotel room"),
]

# --- LangChain Agent Setup ---
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)
llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful hotel booking assistant. Today's date is " + str(datetime.date.today())),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        response = agent_executor.invoke({"input": user_message})
        agent_reply = response.get('output', 'Sorry, I had an issue processing that.')
    except Exception as e:
        print(f"Agent execution error: {e}")
        agent_reply = "An error occurred. Please try again."

    return jsonify({"reply": agent_reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
