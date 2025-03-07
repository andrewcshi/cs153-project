import os
import discord
import logging
import asyncio

from discord.ext import commands
from dotenv import load_dotenv
from agent import MistralAgent

PREFIX = "!"

# Setup terminal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("discord")

# Load the environment variables
load_dotenv('.env')
logger.info("Loaded environment from .env")

# Check if token is available
token = os.getenv("DISCORD_TOKEN")
if not token:
    logger.error("DISCORD_TOKEN not found in environment variables. Make sure your .env file contains the token.")
    exit(1)

# Check if mistral key is available
mistral_key = os.getenv("MISTRAL_API_KEY")
if not mistral_key:
    logger.error("MISTRAL_API_KEY not found in environment variables. Make sure your .env file contains the API key.")
    exit(1)

# Set API keys for Google Maps and Yelp
os.environ["GOOGLE_MAPS_API_KEY"] = os.getenv("GOOGLE_MAPS_API_KEY")
os.environ["YELP_API_KEY"] = os.getenv("YELP_API_KEY")
    
# Create the bot with all intents
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Remove the default help command
bot.remove_command('help')

# Import the Mistral agent from the agent.py file
agent = MistralAgent()

@bot.event
async def on_ready():
    """
    Called when the client is done preparing the data received from Discord.
    Prints message on terminal when bot successfully connects to discord.
    """
    logger.info(f"{bot.user} has connected to Discord!")
    await bot.change_presence(activity=discord.Game(name="Planning your next trip! Type !plan to start."))

@bot.event
async def on_message(message: discord.Message):
    """
    Called when a message is sent in any channel the bot can see.
    """
    # Process commands
    await bot.process_commands(message)

    # Ignore messages from self or other bots to prevent infinite loops
    # Also ignore messages that start with the command prefix
    if message.author.bot or message.content.startswith(PREFIX):
        return

    # Show typing indicator to let user know bot is processing
    async with message.channel.typing():
        try:
            logger.info(f"Processing message from {message.author}: {message.content}")
            
            # Process the message with the agent
            response = await agent.run(message)
            
            # Split response if it's too long for a single Discord message
            if len(response) <= 2000:
                await message.reply(response)
            else:
                # Split into chunks of 2000 characters (Discord's message limit)
                chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                for chunk in chunks:
                    await message.channel.send(chunk)
                    # Brief delay to maintain message order
                    await asyncio.sleep(0.5)
                    
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await message.reply("Sorry, I encountered an error while processing your request. Please try again later.")


@bot.command(name="help", help="Shows the travel planner help message.")
async def help_command(ctx):
    help_text = """
**🌍 TravelBuddy Help 🧳**

I'm your AI travel planning assistant! Here's how I can help you plan your next adventure:

- Suggest destinations based on your interests, budget, and schedule
- Recommend attractions, activities, and local experiences
- Provide transportation and accommodation information
- Create personalized itineraries
- Offer tips on budgeting, packing, and travel safety

**Planning Process:**
1. First, I'll ask about your destination(s)
2. Then, your travel dates
3. Next, your travel preferences (luxury, adventure, etc.)
4. Whether you want accommodation recommendations
5. Finally, your food preferences

**Commands:**
`!plan` - Start a new travel planning session
`!clear` - Clear your conversation history
`!help` - Show this help message

**Example questions:**
• "I want to plan a 5-day trip to Japan in October. What should I see?"
• "What are some budget-friendly beach destinations in Europe?"
• "Help me create an itinerary for a family trip to Orlando"
• "What should I pack for a hiking trip in Colorado in spring?"

*Note: I remember our conversation history to provide better assistance throughout your planning process.*
"""
    await ctx.send(help_text)


@bot.command(name="clear", help="Clears your conversation history with the bot.")
async def clear_history(ctx):
    user_id = str(ctx.author.id)
    if user_id in agent.conversation_history:
        agent.conversation_history[user_id] = []
        # Also reset travel data
        if user_id in agent.travel_data:
            agent.travel_data[user_id] = {
                "stage": 0,  # initial stage
                "locations": [],
                "dates": {},
                "preferences": [],
                "accommodation": {},
                "food": {},
                "itinerary": {}
            }
        await ctx.send("Your conversation history has been cleared! Let's start fresh with your travel plans.")
    else:
        await ctx.send("You don't have any conversation history yet.")


@bot.command(name="plan", help="Start a new travel plan.")
async def start_plan(ctx):
    """Start a new travel planning session."""
    user_id = str(ctx.author.id)
    
    # Clear any existing conversation
    agent.conversation_history[user_id] = []
    
    # Reset travel data
    agent.travel_data[user_id] = {
        "stage": 0,  # INITIAL stage
        "locations": [],
        "dates": {},
        "preferences": [],
        "accommodation": {},
        "food": {},
        "itinerary": {}
    }
    
    # Start with the initial prompt
    initial_prompt = """
**🌍 Let's Plan Your Trip! 🧳**

I'll help you create a personalized travel itinerary. To get started, I'll need some information:

1️⃣ What location(s) are you interested in visiting? (You can list multiple places)
"""
    await ctx.send(initial_prompt)


# Start the bot, connecting it to the gateway
logger.info("Starting bot...")
bot.run(token)