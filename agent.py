import os
from mistralai import Mistral
import discord
from collections import defaultdict

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = """You are TravelBuddy, an expert travel planning assistant.

Your purpose is to help users plan their trips by:
- Suggesting destinations based on their interests, budget, and time constraints
- Recommending attractions, activities, and local experiences
- Providing information about transportation options, accommodations, and local customs
- Creating personalized itineraries
- Offering tips on budgeting, packing, and travel safety

Always be friendly, enthusiastic, and conversational. Ask clarifying questions when needed, but don't overwhelm the user with too many questions at once. 

If you don't know specific details about a destination, be honest about it rather than making up information. Focus on providing practical, actionable advice that helps the user create a memorable trip.
"""

# Maximum number of message pairs to store in history
MAX_HISTORY = 10

class MistralAgent:
    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        
        # Dictionary to store conversation history for each user
        # The key is the user ID, and the value is a list of message dictionaries
        self.conversation_history = defaultdict(list)

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

    async def run(self, message: discord.Message):
        """Process a message and generate a response using conversation history."""
        user_id = str(message.author.id)
        user_content = message.content
        
        # Add the user message to history
        self.add_to_history(user_id, "user", user_content)
        
        # Construct messages with system prompt and history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.get_history(user_id))
        
        # Get response from Mistral
        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )
        
        assistant_content = response.choices[0].message.content
        
        # Add the assistant's response to history
        self.add_to_history(user_id, "assistant", assistant_content)
        
        return assistant_content