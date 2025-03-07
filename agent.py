import os
import json
from mistralai import Mistral
import discord
from collections import defaultdict
import asyncio

# Import our API services
from google_maps_service import GoogleMapsService
from yelp_service import YelpService

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = """You are TravelBuddy, an expert travel planning assistant.

Your purpose is to help users plan their trips by:
- Suggesting destinations based on their interests, budget, and time constraints
- Recommending attractions, activities, and local experiences
- Providing information about transportation options, accommodations, and local customs
- Creating personalized itineraries
- Offering tips on budgeting, packing, and travel safety

Follow this structured approach to help users plan their travel:
1. Ask for the location(s) they want to visit (there may be multiple)
2. Ask for the travel dates
3. Ask about travel preferences (luxury, outdoor adventure, family-friendly, etc.)
4. Ask if they want hotel/accommodation recommendations
5. Ask about food preferences (cuisine types, price range)

Only move to the next question when you have a clear answer for the current one.
If the user has already provided information for future questions in their initial message, acknowledge it and continue with questions they haven't answered yet.

You have access to the Google Maps API and Yelp API to provide accurate information about locations, attractions, hotels, and restaurants.

When creating itineraries, be specific and detailed. Include:
- Daily activities with approximate timing
- Restaurant recommendations for meals
- Transportation suggestions between locations
- Interesting facts about recommended attractions
- Local customs or etiquette tips when relevant

Always be friendly, enthusiastic, and conversational. Ask clarifying questions when needed, but don't overwhelm the user with too many questions at once.

If you don't know specific details about a destination, be honest about it rather than making up information. Focus on providing practical, actionable advice that helps the user create a memorable trip.
"""

# Maximum number of message pairs to store in history
MAX_HISTORY = 10

# Travel planning stages
PLANNING_STAGES = {
    "INITIAL": 0,
    "LOCATION": 1,
    "DATES": 2,
    "PREFERENCES": 3,
    "ACCOMMODATION": 4,
    "FOOD": 5,
    "ITINERARY": 6
}

class MistralAgent:
    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
        YELP_API_KEY = os.getenv("YELP_API_KEY")
        
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        
        # Initialize API services
        try:
            self.google_maps = GoogleMapsService(api_key=GOOGLE_MAPS_API_KEY)
            print("Google Maps API initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Google Maps API: {e}")
            self.google_maps = None
            
        try:
            self.yelp = YelpService(api_key=YELP_API_KEY)
            print("Yelp API initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Yelp API: {e}")
            self.yelp = None
        
        # Dictionary to store conversation history for each user
        self.conversation_history = defaultdict(list)
        
        # Dictionary to store travel planning data for each user
        self.travel_data = defaultdict(lambda: {
            "stage": PLANNING_STAGES["INITIAL"],
            "locations": [],
            "dates": {},
            "preferences": [],
            "accommodation": {},
            "food": {},
            "itinerary": {}
        })
    
    def get_history(self, user_id):
        """Get the conversation history for a specific user."""
        return self.conversation_history.get(user_id, [])
    
    def add_to_history(self, user_id, role, content):
        """Add a message to the user's conversation history."""
        history = self.conversation_history.setdefault(user_id, [])
        history.append({"role": role, "content": content})
        
        # Keep only the MAX_HISTORY most recent message pairs
        if len(history) > MAX_HISTORY * 2:  # Each pair is a user message and an assistant response
            # Remove the oldest pair (first two messages)
            self.conversation_history[user_id] = history[2:]
    
    async def get_location_info(self, location):
        """Get information about a location using Google Maps."""
        if not self.google_maps:
            return {"error": "Google Maps API not available"}
        
        try:
            return self.google_maps.geocode(location)
        except Exception as e:
            return {"error": str(e)}
    
    async def get_attractions(self, location):
        """Get attractions for a location using Google Maps."""
        if not self.google_maps:
            return {"error": "Google Maps API not available"}
        
        try:
            return self.google_maps.get_attractions(location)
        except Exception as e:
            return {"error": str(e)}
    
    async def get_hotels(self, location):
        """Get hotels for a location using Google Maps."""
        if not self.google_maps:
            return {"error": "Google Maps API not available"}
        
        try:
            return self.google_maps.get_hotels(location)
        except Exception as e:
            return {"error": str(e)}
    
    async def get_restaurants(self, location, cuisine=None, price=None):
        """Get restaurants for a location using Yelp."""
        if not self.yelp:
            return {"error": "Yelp API not available"}
        
        try:
            return self.yelp.get_restaurants(location, cuisine=cuisine, price=price)
        except Exception as e:
            return {"error": str(e)}
    
    async def get_activities(self, location, category=None):
        """Get activities for a location using Yelp."""
        if not self.yelp:
            return {"error": "Yelp API not available"}
        
        try:
            return self.yelp.get_activities(location, category=category)
        except Exception as e:
            return {"error": str(e)}
    
    async def enhance_response_with_api_data(self, user_id, response):
        """Enhance the bot's response with real API data."""
        user_data = self.travel_data[user_id]
        stage = user_data["stage"]
        
        # Only enhance when we're at the itinerary stage
        if stage != PLANNING_STAGES["ITINERARY"]:
            return response
        
        # If we have locations, enrich with real data
        if user_data["locations"]:
            main_location = user_data["locations"][0]
            
            # Gather API data in parallel
            api_tasks = [
                self.get_attractions(main_location),
                self.get_hotels(main_location),
                self.get_restaurants(main_location, 
                                    cuisine=user_data["food"].get("cuisine"),
                                    price=user_data["food"].get("price"))
            ]
            
            api_results = await asyncio.gather(*api_tasks)
            attractions, hotels, restaurants = api_results
            
            # Create a data context for the LLM to use
            data_context = {
                "attractions": attractions.get("results", [])[:5] if "results" in attractions else [],
                "hotels": hotels.get("results", [])[:5] if "results" in hotels else [],
                "restaurants": restaurants.get("businesses", [])[:5] if "businesses" in restaurants else []
            }
            
            # Create a message with the API data
            api_data_message = f"""
Here is real data from APIs to help with recommendations:

ATTRACTIONS:
{json.dumps([{
    "name": a.get("name", "Unknown"),
    "rating": a.get("rating", "N/A"),
    "address": a.get("vicinity", "N/A")
} for a in data_context["attractions"]], indent=2)}

HOTELS:
{json.dumps([{
    "name": h.get("name", "Unknown"),
    "rating": h.get("rating", "N/A"),
    "address": h.get("vicinity", "N/A")
} for h in data_context["hotels"]], indent=2)}

RESTAURANTS:
{json.dumps([{
    "name": r.get("name", "Unknown"),
    "rating": r.get("rating", "N/A"),
    "price": r.get("price", "N/A"),
    "cuisine": [c.get("title") for c in r.get("categories", [])]
} for r in data_context["restaurants"]], indent=2)}

Use this real data to enhance your itinerary recommendations. Include specific attraction names, hotels, and restaurants in your response.
"""
            
            # Get an enhanced response using the API data
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": api_data_message},
                {"role": "user", "content": f"Here's my travel preferences: {json.dumps(user_data, indent=2)}"},
                {"role": "assistant", "content": response},
                {"role": "user", "content": "Can you enhance this itinerary with the specific attractions, hotels, and restaurants from the API data?"}
            ]
            
            try:
                enhanced_response = await self.client.chat.complete_async(
                    model=MISTRAL_MODEL,
                    messages=messages,
                )
                return enhanced_response.choices[0].message.content
            except Exception as e:
                print(f"Error enhancing response: {e}")
                return response
        
        return response
    
    async def run(self, message: discord.Message):
        """Process a message and generate a response using conversation history."""
        user_id = str(message.author.id)
        user_content = message.content
        
        # Add the user message to history
        self.add_to_history(user_id, "user", user_content)
        
        # Update user travel data
        self.extract_travel_information(user_id, user_content)
        
        # Create contextual prompt based on travel data
        context_prompt = self.get_context_prompt(user_id)
        
        # Construct messages with system prompt, history and context
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": context_prompt}
        ]
        messages.extend(self.get_history(user_id))
        
        # Get response from Mistral
        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )
        
        assistant_content = response.choices[0].message.content
        
        # Check if we should enhance the response with API data
        enhanced_content = await self.enhance_response_with_api_data(user_id, assistant_content)
        
        # Add the assistant's response to history
        self.add_to_history(user_id, "assistant", enhanced_content)
        
        return enhanced_content
    
    def extract_travel_information(self, user_id, message):
        """Extract travel information from user message and update travel data."""
        user_data = self.travel_data[user_id]
        current_stage = user_data["stage"]
        
        # TODO: Add more sophisticated extraction logic
        
        # Extract locations
        if current_stage == PLANNING_STAGES["INITIAL"] or current_stage == PLANNING_STAGES["LOCATION"]:
            if "go to" in message.lower() or "visit" in message.lower() or "traveling to" in message.lower():
                # Simple extraction - in reality, you'd want to use NER or similar
                user_data["stage"] = PLANNING_STAGES["DATES"]
                
        # Extract dates
        elif current_stage == PLANNING_STAGES["DATES"]:
            if "from" in message.lower() and "to" in message.lower() or "between" in message.lower():
                user_data["stage"] = PLANNING_STAGES["PREFERENCES"]
                
        # Extract preferences
        elif current_stage == PLANNING_STAGES["PREFERENCES"]:
            if "luxury" in message.lower() or "adventure" in message.lower() or "family" in message.lower():
                user_data["stage"] = PLANNING_STAGES["ACCOMMODATION"]
                
        # Extract accommodation preferences
        elif current_stage == PLANNING_STAGES["ACCOMMODATION"]:
            if "hotel" in message.lower() or "airbnb" in message.lower() or "hostel" in message.lower():
                user_data["stage"] = PLANNING_STAGES["FOOD"]
                
        # Extract food preferences
        elif current_stage == PLANNING_STAGES["FOOD"]:
            if "restaurant" in message.lower() or "cuisine" in message.lower() or "food" in message.lower():
                user_data["stage"] = PLANNING_STAGES["ITINERARY"]
    
    def get_context_prompt(self, user_id):
        """Get contextual prompt based on current planning stage."""
        user_data = self.travel_data[user_id]
        stage = user_data["stage"]
        
        if stage == PLANNING_STAGES["INITIAL"]:
            return "The user is starting to plan a trip. Ask them about their desired destination(s)."
            
        elif stage == PLANNING_STAGES["LOCATION"]:
            return "The user is choosing destinations. Ask about when they plan to travel (dates)."
            
        elif stage == PLANNING_STAGES["DATES"]:
            return "The user has provided travel dates. Ask about their travel preferences (luxury, adventure, family-friendly, etc.)."
            
        elif stage == PLANNING_STAGES["PREFERENCES"]:
            return "The user has shared their travel preferences. Ask if they want hotel/accommodation recommendations and their preferences."
            
        elif stage == PLANNING_STAGES["ACCOMMODATION"]:
            return "The user has provided accommodation preferences. Ask about their food preferences (cuisines, budget)."
            
        elif stage == PLANNING_STAGES["FOOD"]:
            return "The user has shared their food preferences. Tell them you'll create an itinerary based on all their preferences."
            
        elif stage == PLANNING_STAGES["ITINERARY"]:
            return f"""
The user has provided all necessary information for a travel plan. Create a detailed itinerary for them.

Their preferences:
Locations: {', '.join(user_data['locations']) if user_data['locations'] else 'Not specified'}
Dates: {user_data['dates'].get('raw', 'Not specified')}  
Travel Preferences: {', '.join(user_data['preferences']) if user_data['preferences'] else 'Not specified'}
Accommodation: {user_data['accommodation'].get('preferences', 'Not specified')}
Food: {user_data['food'].get('preferences', 'Not specified')}

Create a detailed day-by-day itinerary with specific recommendations for attractions, hotels, and restaurants.
"""
        
        return "Continue helping the user plan their trip."