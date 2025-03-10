import os
import json
from mistralai import Mistral
import discord
from collections import defaultdict
import asyncio
import re

# Import our API services
from google_maps_service import GoogleMapsService
from yelp_service import YelpService
from button_utils import send_buttons_message

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
------------------------------------------------------------
API DATA FOR ITINERARY ENRICHMENT
------------------------------------------------------------
Below is the latest data from our APIs. Integrate these details into the itinerary you generate. When including this data, ensure that your output follows the template exactly.

------------------------------------------------------------
Attraction Data:
{json.dumps([{
    "name": a.get("name", "Unknown"),
    "rating": a.get("rating", "N/A"),
    "address": a.get("vicinity", "N/A")
} for a in data_context["attractions"]], indent=2)}

------------------------------------------------------------
Hotel Data:
{json.dumps([{
    "name": h.get("name", "Unknown"),
    "rating": h.get("rating", "N/A"),
    "address": h.get("vicinity", "N/A")
} for h in data_context["hotels"]], indent=2)}

------------------------------------------------------------
Restaurant Data:
{json.dumps([{
    "name": r.get("name", "Unknown"),
    "rating": r.get("rating", "N/A"),
    "price": r.get("price", "N/A"),
    "cuisine": [c.get("title") for c in r.get("categories", [])]
} for r in data_context["restaurants"]], indent=2)}

------------------------------------------------------------
When updating the itinerary:
- Follow the exact output template provided in the itinerary prompt.
- Insert API data by selecting specific names, ratings, and addresses into the relevant sections (e.g., under **Attraction:** or **Hotel:**).
- Ensure your final itinerary output includes the horizontal dividers and clear labels as specified in the template.

Incorporate these details to produce a structured and visually engaging itinerary.
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
    
    def get_button_options(self, user_id, response):
        """
        Extract potential button options from the response based on the current planning stage.
        
        Args:
            user_id: The user ID
            response: The response from the LLM
            
        Returns:
            A tuple of (modified_response, button_options) if buttons should be shown,
            or (original_response, None) if no buttons are needed
        """
        user_data = self.travel_data[user_id]
        current_stage = user_data["stage"]
        
        # Predefined options for each stage
        predefined_options = {
            PLANNING_STAGES["INITIAL"]: [],  # Handled in bot.py
            PLANNING_STAGES["LOCATION"]: [],  # Handled in bot.py
            PLANNING_STAGES["DATES"]: [
                "Next week", 
                "Next month", 
                "This summer", 
                "This winter", 
                "Flexible"
            ],
            PLANNING_STAGES["PREFERENCES"]: [
                "Luxury", 
                "Budget-friendly", 
                "Adventure", 
                "Cultural", 
                "Family-friendly"
            ],
            PLANNING_STAGES["ACCOMMODATION"]: [
                "Hotel", 
                "Airbnb", 
                "Resort", 
                "Hostel", 
                "Vacation rental"
            ],
            PLANNING_STAGES["FOOD"]: [
                "Local cuisine", 
                "Fine dining", 
                "Street food", 
                "Vegetarian", 
                "All options"
            ]
        }
        
        # Define patterns to look for options in the response
        option_patterns = {
            PLANNING_STAGES["PREFERENCES"]: [
                r"(?:luxury|budget|mid-range|affordable|high-end)",
                r"(?:adventure|relaxation|cultural|family-friendly|romantic|solo|group)",
                r"(?:outdoor|indoor|sightseeing|shopping|nightlife|food|history)"
            ],
            PLANNING_STAGES["ACCOMMODATION"]: [
                r"(?:hotel|hostel|resort|airbnb|vacation rental|bed and breakfast|camping)"
            ],
            PLANNING_STAGES["FOOD"]: [
                r"(?:local cuisine|international|fast food|fine dining|street food|vegetarian|vegan|seafood)"
            ]
        }
        
        # Check if we should add buttons based on the current stage
        if current_stage in predefined_options and predefined_options[current_stage]:
            # Use predefined options for this stage
            options = predefined_options[current_stage]
            modified_response = response + "\n\n*Click a button below to quickly select an option:*"
            return modified_response, options
        elif current_stage in option_patterns:
            # Try to extract options from the response
            options = []
            for pattern in option_patterns[current_stage]:
                matches = re.findall(pattern, response.lower())
                options.extend(matches)
            
            # Remove duplicates and limit to 5 options (Discord UI constraint)
            options = list(set(options))[:5]
            
            # Capitalize options for better presentation
            options = [opt.capitalize() for opt in options]
            
            if options:
                # Add a note about button options to the response
                modified_response = response + "\n\n*Click a button below to quickly select an option:*"
                return modified_response, options
        
        # Default: no buttons
        return response, None
    
    async def process_button_selection(self, interaction, option, user_id, channel):
        """Process a button selection and generate a response."""
        # Add the user's selection to history
        self.add_to_history(user_id, "user", option)
        
        # Update travel data based on the selection
        self.extract_travel_information(user_id, option)
        
        # Generate a response to the selection
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
        
        # Check if we should add button options
        modified_content, new_button_options = self.get_button_options(user_id, enhanced_content)
        
        # Add the assistant's response to history
        self.add_to_history(user_id, "assistant", enhanced_content)
        
        # Get the actual channel object if we have an interaction
        if hasattr(interaction, 'channel') and interaction.channel:
            channel = interaction.channel
        
        # Send the response
        if new_button_options:
            # Define a new callback for the buttons in this response
            async def next_button_callback(interaction, option):
                await self.process_button_selection(interaction, option, user_id, channel)
            
            await send_buttons_message(channel, modified_content, new_button_options, next_button_callback)
        else:
            await channel.send(modified_content)
    
    async def run(self, message: discord.Message):
        user_id = str(message.author.id)
        user_content = message.content

        # Optional: If the user is at the ITINERARY stage, clear the conversation history
        # to ensure that the new strict prompt is not diluted by older messages.
        if self.travel_data[user_id]["stage"] == PLANNING_STAGES["ITINERARY"]:
            self.conversation_history[user_id] = []  # reset history for a fresh context

        # Add the user message to history
        self.add_to_history(user_id, "user", user_content)

        # Update user travel data
        self.extract_travel_information(user_id, user_content)

        # Create contextual prompt based on travel data
        context_prompt = self.get_context_prompt(user_id)

        # Construct messages with system prompt, extra instructions, and full (or truncated) history
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": "Below is the complete conversation history to help structure the final itinerary:"},
            {"role": "system", "content": context_prompt}
        ]
        # Optionally, truncate history if it's too long
        full_history = self.get_history(user_id)
        if len(full_history) > MAX_HISTORY * 2:
            full_history = full_history[-(MAX_HISTORY * 2):]
        messages.extend(full_history)

        # Get response from Mistral
        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )

        assistant_content = response.choices[0].message.content

        # Enhance the response with API data if in ITINERARY stage
        enhanced_content = await self.enhance_response_with_api_data(user_id, assistant_content)

        # Add the assistant's (final) response to the conversation history
        self.add_to_history(user_id, "assistant", enhanced_content)

        # Check if we should add button options and send accordingly
        modified_content, button_options = self.get_button_options(user_id, enhanced_content)
        if button_options:
            async def button_callback(interaction, option):
                await self.process_button_selection(interaction, option, user_id, message.channel)
            await send_buttons_message(message, modified_content, button_options, button_callback)
            return ""  # Return empty string since message has been sent

        return enhanced_content

    def extract_travel_information(self, user_id, message):
        """Extract travel information from user message and update travel data."""
        user_data = self.travel_data[user_id]
        current_stage = user_data["stage"]
        
        # Check for location information
        if current_stage == PLANNING_STAGES["INITIAL"] or current_stage == PLANNING_STAGES["LOCATION"]:
            location_indicators = ["visit", "go to", "travel to", "traveling to", "destination", "want to see"]
            
            # Check if the message contains location information
            if any(indicator in message.lower() for indicator in location_indicators):
                # Simple extraction - in a real implementation, you'd want to use NER
                user_data["stage"] = PLANNING_STAGES["DATES"]
                
                # Try to extract locations (very basic implementation)
                potential_locations = re.findall(r'(?:visit|go to|travel to|traveling to|want to see)\s+([A-Za-z\s,]+)', message.lower())
                if potential_locations:
                    # Clean up the extracted locations
                    locations = [loc.strip() for loc in potential_locations[0].split(',')]
                    user_data["locations"].extend(locations)
        
        # Check for date information
        elif current_stage == PLANNING_STAGES["DATES"]:
            date_indicators = ["from", "to", "between", "during", "in", "next week", "next month", "this summer", "this winter", "flexible"]
            
            if any(indicator in message.lower() for indicator in date_indicators):
                user_data["stage"] = PLANNING_STAGES["PREFERENCES"]
                
                # Store the date information (simplified)
                user_data["dates"]["text"] = message
        
        # Check for preference information
        elif current_stage == PLANNING_STAGES["PREFERENCES"]:
            preference_indicators = ["luxury", "budget", "adventure", "cultural", "family", "romantic", "solo", "group"]
            
            if any(indicator in message.lower() for indicator in preference_indicators):
                user_data["stage"] = PLANNING_STAGES["ACCOMMODATION"]
                
                # Store the preference information
                user_data["preferences"].append(message)
        
        # Check for accommodation information
        elif current_stage == PLANNING_STAGES["ACCOMMODATION"]:
            accommodation_indicators = ["hotel", "hostel", "resort", "airbnb", "vacation rental", "bed and breakfast", "camping"]
            
            if any(indicator in message.lower() for indicator in accommodation_indicators):
                user_data["stage"] = PLANNING_STAGES["FOOD"]
                
                # Store the accommodation preference
                user_data["accommodation"]["preference"] = message
        
        # Check for food information
        elif current_stage == PLANNING_STAGES["FOOD"]:
            food_indicators = ["cuisine", "food", "restaurant", "dining", "vegetarian", "vegan", "local", "international"]
            
            if any(indicator in message.lower() for indicator in food_indicators):
                user_data["stage"] = PLANNING_STAGES["ITINERARY"]
                
                # Store the food preference
                user_data["food"]["preference"] = message
    
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
------------------------------------------------------------
FINAL ITINERARY REQUEST
------------------------------------------------------------
You must generate a detailed day-by-day itinerary using markdown that follows the exact structure below. Do not deviate from this format:

------------------------------------------------------------
[TEMPLATE START]

**DAY [NUMBER]**  
------------------------------------------------------------
**Date:** [Insert Date or Time Block, e.g., "June 15th, Morning"]  
**Activity:** [Insert main activity, e.g., "Arrival and Check-in"]  
**Details:**  
- **Hotel:** [Insert hotel name] - [Insert booking hyperlink if available, e.g., "[Reserve Now](#)"]  
- **Restaurant:** [Insert restaurant name] - [Insert booking hyperlink if available]  
- **Attraction:** [Insert attraction/experience name]  
- **Time:** [Insert time block, e.g., "10:30 AM – 12:00 PM"]  
- **Alternative Option:** [Optional alternative choice]

Repeat this block for each day.

[TEMPLATE END]
------------------------------------------------------------
Formatting Requirements:
- Use **bullet points** for lists.
- Use horizontal lines (a series of dashes "------------------------------------------------------------") to separate sections.
- Use clear labels exactly as shown (e.g., **Hotel:**, **Restaurant:**, **Attraction:**, **Date:**, **Activity:**, **Time:**).
- Do not include any emojis; only use text-based labels.
- Ensure that all booking hyperlinks are included as placeholders (e.g., `[Reserve Now](#)`) where applicable.
- Your final output must strictly follow the template above.

------------------------------------------------------------
User Preferences:
- **Locations:** {', '.join(user_data['locations']) if user_data['locations'] else 'Not specified'}
- **Dates:** {user_data['dates'].get('text', 'Not specified')}
- **Travel Preferences:** {', '.join(user_data['preferences']) if user_data['preferences'] else 'Not specified'}
- **Accommodation Preference:** {user_data['accommodation'].get('preference', 'Not specified')}
- **Food Preference:** {user_data['food'].get('preference', 'Not specified')}

------------------------------------------------------------
Generate the itinerary exactly following the template above, ensuring clear section dividers and all required labels.
"""

        return "Continue helping the user plan their trip."