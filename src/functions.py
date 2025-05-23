# Functions
# Reference: https://github.com/aniketmaurya/agents/blob/main/src/agents/tools.py
import wikipedia
import json
import requests
from langchain_community.utilities import WikipediaAPIWrapper
from googlesearch import search
import geocoder
from datetime import datetime, timedelta
import googlemaps
from gcsa.google_calendar import GoogleCalendar
from gcsa.event import Event
import python_weather
import asyncio
from hk_bus_eta import HKEta
import time
import threading
from openai import AzureOpenAI
from utils import azure_inference
api_key = "4d2ff10a8c3d4d09883a4411832b6718" # Azure API key
client = AzureOpenAI(
    api_key = api_key,  
    api_version = "2023-05-15",
    azure_endpoint = "https://cuhk-aiot-gpt4.openai.azure.com/"
)
wikipedia_api_wrapper = None


def get_current_datetime() -> str:
    """Get the current date and time.
    Args:
        None.
    Returns:
        str: Current date and time.
    """
    now = datetime.now()
    return f"Date: {now.strftime('%B %d, %Y')} Time: {now.strftime('%H:%M:%S')}"

def get_current_gps_coordinates() -> str:
    """Get the current GPS coordinates of the user.
    Args:
        None.
    Returns:
        str: GPS coordinates of the user.
    """
    g = geocoder.ip('me')
    return g.address

def get_city_weather(city: str, time: str) -> str:
    """Get the weather for a specified city at a given time.

    Args:
        city (str): The city to fetch weather for.
        time (str): The time to fetch weather for.

    Returns:
        str: weather condition for a specified city at a given time.
    """
    # async def fetch_weather():
    #     async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
    #         # Fetch a weather forecast from a city
    #         weather = await client.get(city)  # e.g., 'New York'
    #         # Returns the current day's forecast temperature (int)
    #         print(weather.temperature)
    #         # Get the weather forecast for a few days
    #         summary_weather = ''
    #         for daily in weather:
    #             summary_weather += f" --> {daily!r}"
    #         return summary_weather
    # summary_weather = asyncio.run(fetch_weather())

    summary_weather = f"The weather in {city} on {time} be 36 degrees with heavy rain and thunderstorms."
    return summary_weather


def wikipedia_search(query: str) -> str:
    """Tool that searches the Wikipedia API. Useful for when you need to answer
    general questions about people, places, companies, facts, historical
    events, or other subjects.

    Args:
        Input should be a search query.

    Returns:
        Wikipedia search result.
    """
    # global wikipedia_api_wrapper
    # if wikipedia_api_wrapper is None:
    #     wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=1)

    # return wikipedia_api_wrapper.run(query)
    return f"Search results for '{query}' in Wikipedia."

def google_search(query: str) -> str:
    """Performs a Google search for the given query, retrieves the top search
    result URLs and description from the page.

    Args:
        Input should be a search query.

    Returns:
        URL and description of the search result.
    """
    for result in search(query, num_results=1, sleep_interval=5, advanced=True):
        return f"url:{result.url},\ndescription:{result.description}"
    # return f"Search results for '{query}' in Google."


def check_agenda_time_conflict(time: datetime) -> str:
    """
    Check if there is a time conflict in the user's agenda for a given datetime
    and return all events as a summarized string.

    Args:
        time (datetime): The time to check for conflicts.

    Returns:
        str: A summary of all events and whether there is a conflict.
    """
    # calendar = GoogleCalendar('bfyang@yahoo.com')
    # events_summary = []
    # for event in calendar:
    #     event_start = event.start
    #     event_end = event.end if event.end else event.start + timedelta(hours=1)  # Assume 1-hour duration if no end time

    #     # Format event times as strings without timezone information
    #     event_start_str = event_start.strftime('%Y-%m-%d %H:%M:%S')
    #     event_end_str = event_end.strftime('%Y-%m-%d %H:%M:%S')

    #     event_summary = f"Event: {event.summary}, Start: {event_start_str}, End: {event_end_str}"
    #     events_summary.append(event_summary)

    # Combine all events into a single string
    events_str = "\n".join('success')

    return f"Agenda Summary:\n\n{events_str}"


prompt_time_transform = "You are a time transformer. Your task is to convert a user's natural language description of a time into a specific format. You will receive a description from the user, such as \"Friday,\" and the current date and time, like \"2025-04-08 20:37:47.263688.\" Your output should be the exact date and time corresponding to the user's description, formatted as follows: (year, month, day, hour, minute). For example, (2025, 4, 9, 10, 0). Only output this turble without any additional words."
def add_to_agenda(event: str, time: str) -> str:
    """
    Add an event to the user's agenda.

    Args:
        event (str): The name of the event to add.
        time (str): The time of the event.

    Returns:
        str: The confirmation message.
"""

    # calendar = GoogleCalendar('bfyang@yahoo.com')
    # current_time = datetime.now()
    # messages = []
    # messages.append({"role": "system", "content": prompt_time_transform})
    # messages.append({"role": "user", "content": f"Current time is {current_time}. The user description is {time}."})
    # transformed_time = azure_inference(client, 'gpt-4o', messages, temperature=0.7, max_tokens=4096)
    # print("Transformed Datatime: ", transformed_time)

    # # Parse the transformed time
    # parsed_time = eval(transformed_time)
    # event = Event(
    #     event,
    #     start=datetime(*parsed_time),
    #     minutes_before_popup_reminder=15
    # )
    # calendar.add_event(event)

    return f"Event '{event}' added to agenda for {time}."

def get_online_product_price(product_name: str) -> str:
    """
    Get the price of a product from an online store such as Amazon or eBay.

    Args:
        product_name (str): The name of the product to search for.

    Returns:
        str: The price of the product as a string.
    """
    return f"The price of the {product_name} is $600"

def get_health_data():
    """
    Get health data from the user's smart device.

    Args:
        None.

    Returns:
        str: The health data as a string.
    """
    # See: https://python-fitbit.readthedocs.io/en/latest/

    return "The user's heart rate is 70 bpm, oxygen saturation is 98%, respiration rate is 16 bpm, body temperature is 36.5 degrees Celsius, total steps today is 5000, and total calories burned today is 2000."

def get_medical_knowledge(query: str) -> str:
    """
    Get medical expert knowledge from the up-to-date medical knowledge database.

    Args:
        query (str): The query string containing the medical topic or symptoms.

    Returns:
        str: The medical expert knowledge as a string.
    """
    return "The symptoms of COVID-19 include fever, cough, shortness of breath, fatigue, body aches, loss of taste or smell, sore throat, and headache. If you experience any of these symptoms, please contact your healthcare provider immediately."
    
def vllm(prompt: str, image_path: str) -> str:
    """
    Visual Large Language Model (VLM) that can answer the user's questions based on the given image.
    Args:
        prompt (str): The prompt containing the user's question.
        image_path (str): The path to the image file.
    Returns:
        str: The response from the VLLM.
    """
    return None

GOOGLE_MAP_API_KEY = 'AIzaSyBRmEfw53J51oF-N7no2auaijGOCTyRJTE'
def google_map(current_location: str, destination: str) -> str:
    """
    Get the route and distance from the current location to the destination using Google Maps API.

    Args:
        current_location (str): The starting location.
        destination (str): The destination location.

    Returns:
        str: The route and distance information.
    """
    # gmaps = googlemaps.Client(GOOGLE_MAP_API_KEY)
    # current_location = get_current_gps_coordinates()
    # response = gmaps.distance_matrix(current_location, destination, mode = 'driving')
    # print(response)
    # return response

    return f"The distance from {current_location} to {destination} is 2 km."

def book_uber(current_location: str, destination: str) -> str:
    """
    Book an Uber ride from the current location to the destination.

    Args:
        current_location (str): The starting location.
        destination (str): The destination location.

    Returns:
        str: The Uber ride booking confirmation.
    """
    return f"Uber ride booked from {current_location} to {destination}. It will takes 15 minutes to arrive and 30 dollars."

def play_music() -> str:
    """
    Play a song from the user's music library.

    Args:
        None.

    Returns:
        str: The song playing confirmation.
    """
    return f"Now playing music."

def check_bus_schedule(bus_stop: str) -> str:
    """
    Check the bus schedule for a specific bus stop.

    Args:
        bus_stop (str): The name of the bus stop.

    Returns:
        str: The bus schedule information.
    """
    # hketa = HKEta()
    # etas = hketa.getEtas(route_id = "TCL+1+Hong Kong+Tung Chung", seq=0, language="en")
    # return etas
    return f"The next bus at {bus_stop} is at 9:30 AM."


def timer_thread(seconds: int):
    time.sleep(seconds)
    print("Time is up!")

prompt_timer_transform = "You are a time transformer. Your task is to convert a user's natural language description of a time into a time in a specific format. You will receive a description from the user, such as \"30 seconds,\"  Your output should be the exact time in seconds. For example, 30. Only output this number without any additional words."
def set_timer(duration: str) -> str:
    """
    Set a timer for a specific duration.

    Args:
        duration (str): The duration of the timer.

    Returns:
        str: The timer set confirmation.
    """
    # messages = []
    # messages.append({"role": "system", "content": prompt_timer_transform})
    # messages.append({"role": "user", "content": duration})
    # transformed_time = azure_inference(client, 'gpt-4o', messages, temperature=0.7, max_tokens=4096)
    # seconds = int(transformed_time)
    # thread = threading.Thread(target=timer_thread, args=(seconds,))
    # thread.start()

    return f"Timer set for {duration}."

def search_rednote(query: str) -> str:
    """
    A platform where people share tips on travel, fitness, cooking, and more. You can use it to search for relevant strategies there.

    Args:
        query (str): The search query.

    Returns:
        str: The search results from rednote.
    """
    # See: https://apify.com/easyapi/rednote-xiaohongshu-comments-scraper/api/python
    return f"Search results for '{query}' in rednote."


def add_meeting(meeting_topic: str, start_time: str, meeting_location: str) -> str:
    """
    This API allows users to make a reservation for a meeting and store the meeting information (e.g., topic, time, location, attendees) in the database.

    Args:
        meeting_topic (str): The topic of the meeting.
        start_time (str): The start time of the meeting, in the pattern of %Y-%m-%d %H:%M:%S'.
        meeting_location (str): The location where the meeting to be held.

    Returns:
        status (str): success or failed.
    """
    return f"success"

def send_email(receiver: str, subject: str, content: str) -> str:
    """
    This API for sending email, given the receiver, subject and content.

    Args:
        receiver (str): The receiver address of the email.
        subject (str): The subject address of the email.
        content (str): The content of the email.

    Returns:
        status (str): The status of the email.
    """
    return f"success"

def query_stock(stock_code, date):
    """
    This API queries the stock price of a given stock code and date.

    Args:
        stock_code (str): The stock code of the given stock.
        date (str): The date of the stock price. Format: %Y-%m-%d.

    Returns:
        stock_price (str): The stock price of the given stock.
    """
    return f"success"


functions = {
    "get_city_weather": get_city_weather,
    "get_current_datetime": get_current_datetime,
    "get_current_gps_coordinates": get_current_gps_coordinates,
    "wikipedia_search": wikipedia_search,
    "google_search": google_search,
    "check_agenda_time_conflict": check_agenda_time_conflict,
    "get_online_product_price": get_online_product_price,
    "get_health_data": get_health_data,
    "get_medical_knowledge": get_medical_knowledge,
    "vllm": vllm,
    "google_map": google_map,
    "book_uber": book_uber,
    "play_music": play_music,
    "add_to_agenda": add_to_agenda,
    "check_bus_schedule": check_bus_schedule,
    "set_timer": set_timer,
    "search_rednote": search_rednote,
    "add_meeting": add_meeting,
    "send_email": send_email,
    "query_stock": query_stock,
    }

def process_function_call(json_tool):
    # Reference:
    # https://github.com/mshojaei77/simple_function_calling/blob/main/main.py#L41
    # https://dev.to/angu10/a-step-by-step-guide-to-llm-function-calling-in-python-4pg7
    try:
        params = json_tool['parameters']
        # Call the appropriate function
        if json_tool['name'] in functions:
            # func
            fun = functions.get(json_tool['name'])
            # execute the function
            if params == 'None':
                result = fun()
            else:
                result = fun(**params)
            return json.dumps(result)
        else:
            raise NotImplementedError(f"Function {json_tool['name']} is not implemented.")
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    pass

    # # Google Map
    # google_map("The Chinese University of Hong Kong", "Mong Kok")

    # # Check Calendar
    # current_time = datetime.now()
    # agenda_summary = check_agenda_time_conflict(current_time)
    # print(agenda_summary)

    # # Add agenda
    # add_to_agenda("Group meeting", "This Friday at 3 PM")

    # # Get city weather
    # summary_weather = get_city_weather("New York", "12:00 PM")
    # print(summary_weather)

    # # Check bus schedule
    # bus_schedule = check_bus_schedule("Tung Chung Station")
    # print(bus_schedule)

    # # Set timer
    # duration = 5
    # set_timer(duration)

    # # GPS coordinates
    # gps_coordinates = get_current_gps_coordinates()
    # print(gps_coordinates)