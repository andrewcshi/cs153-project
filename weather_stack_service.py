import os
import requests
from datetime import datetime, timedelta


class WeatherStackService:
    BASE_URL = "http://api.weatherstack.com"

    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.getenv("WEATHER_STACK_API_KEY")

        if not api_key:
            raise ValueError("WeatherStack API key is required")

        self.api_key = api_key

    def get_current_weather(self, location):
        """Get current weather for a location."""
        endpoint = f"{self.BASE_URL}/current"
        params = {
            "access_key": self.api_key,
            "query": location,
            "units": "m",  # Metric units
        }

        response = requests.get(endpoint, params=params)
        return response.json()

    def get_historical_weather(self, location, date):
        """Get historical weather for a location on a specific date."""
        endpoint = f"{self.BASE_URL}/historical"

        # Format date as YYYY-MM-DD
        if isinstance(date, datetime):
            date_str = date.strftime("%Y-%m-%d")
        else:
            date_str = date

        params = {
            "access_key": self.api_key,
            "query": location,
            "historical_date": date_str,
            "units": "m",  # Metric units
        }

        response = requests.get(endpoint, params=params)
        return response.json()

    def analyze_best_travel_dates(
        self, location, start_date=None, end_date=None, preferences=None
    ):
        """
        Analyze weather to recommend travel dates.

        Parameters:
        - location: Location name (e.g., "Paris")
        - start_date: Start of possible travel period (datetime or string YYYY-MM-DD)
        - end_date: End of possible travel period (datetime or string YYYY-MM-DD)
        - preferences: Dictionary of weather preferences (e.g., {"temperature": "warm", "precipitation": "low"})

        Returns:
        - Dictionary with recommendation information
        """
        # Get current weather as baseline
        current = self.get_current_weather(location)

        if "current" not in current:
            return {
                "recommendation": "Weather data unavailable for this location",
                "error": current.get("error", {}).get("info", "Unknown error"),
            }

        # Extract current conditions
        temp = current["current"].get("temperature", 0)
        weather_desc = current["current"].get("weather_descriptions", ["Unknown"])[0]
        is_sunny = "sunny" in weather_desc.lower() or "clear" in weather_desc.lower()
        is_rainy = "rain" in weather_desc.lower() or "shower" in weather_desc.lower()

        # Analyze based on preferences
        if preferences:
            temp_pref = preferences.get("temperature", "moderate")
            precip_pref = preferences.get("precipitation", "any")

            # Simple recommendation logic
            if temp_pref == "warm" and temp < 15:
                recommendation = "Current temperatures are cooler than your preference. Consider delaying your trip if possible for warmer weather."
            elif temp_pref == "cool" and temp > 25:
                recommendation = "Current temperatures are warmer than your preference. Consider scheduling your trip during a cooler season."
            elif precip_pref == "low" and is_rainy:
                recommendation = "There's currently precipitation in the area. Check the forecast for your travel dates."
            elif is_sunny and temp >= 15 and temp <= 25:
                recommendation = "Current weather conditions are ideal! This is a great time to visit."
            else:
                recommendation = f"Current conditions: {weather_desc}, {temp}°C. Check specific dates for more accurate forecasts."
        else:
            # Default recommendation
            if is_sunny and not is_rainy and temp >= 15 and temp <= 25:
                recommendation = (
                    "Current weather conditions are pleasant and ideal for sightseeing."
                )
            else:
                recommendation = f"Current conditions: {weather_desc}, {temp}°C. Consider your weather preferences when planning."

        return {
            "current_weather": current["current"],
            "recommendation": recommendation,
            "location": current.get("location", {}).get("name", location),
        }

    def get_weather_description(self, weather_data):
        """Convert weather data to a human-readable description."""
        if "current" not in weather_data:
            return "Weather information is unavailable."

        current = weather_data["current"]
        location = weather_data.get("location", {})

        temp = current.get("temperature", "N/A")
        feels_like = current.get("feelslike", "N/A")
        description = current.get("weather_descriptions", ["Unknown"])[0]
        humidity = current.get("humidity", "N/A")
        precip = current.get("precip", "N/A")
        wind_speed = current.get("wind_speed", "N/A")

        location_name = location.get("name", "the requested location")

        return (
            f"Current conditions in {location_name}: {description}, {temp}°C (feels like {feels_like}°C). "
            f"Humidity: {humidity}%, Precipitation: {precip}mm, Wind speed: {wind_speed} km/h."
        )
