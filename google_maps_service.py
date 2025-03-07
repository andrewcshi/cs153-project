import os
import requests

class GoogleMapsService:
    BASE_URL = "https://maps.googleapis.com/maps/api"
    
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        
        if not api_key:
            raise ValueError("Google Maps API key is required")
        
        self.api_key = api_key
    
    def geocode(self, address):
        """Convert an address to coordinates."""
        endpoint = f"{self.BASE_URL}/geocode/json"
        params = {
            "address": address,
            "key": self.api_key
        }
        response = requests.get(endpoint, params=params)
        return response.json()
    
    def place_search(self, query, location=None, radius=None, type=None):
        """Search for places using text search."""
        endpoint = f"{self.BASE_URL}/place/textsearch/json"
        
        params = {
            "query": query,
            "key": self.api_key
        }
        
        if location:
            params["location"] = location
        if radius:
            params["radius"] = radius
        if type:
            params["type"] = type
            
        response = requests.get(endpoint, params=params)
        return response.json()
    
    def place_details(self, place_id):
        """Get detailed information about a place."""
        endpoint = f"{self.BASE_URL}/place/details/json"
        params = {
            "place_id": place_id,
            "key": self.api_key,
            "fields": "name,formatted_address,rating,formatted_phone_number,website,opening_hours,price_level,photos"
        }
        response = requests.get(endpoint, params=params)
        return response.json()
    
    def get_attractions(self, location, radius=5000):
        """Get tourist attractions near a location."""
        endpoint = f"{self.BASE_URL}/place/nearbysearch/json"
        
        # First geocode the location to get coordinates
        geocode_result = self.geocode(location)
        if not geocode_result.get("results"):
            return {"error": "Could not geocode location"}
        
        lat = geocode_result["results"][0]["geometry"]["location"]["lat"]
        lng = geocode_result["results"][0]["geometry"]["location"]["lng"]
        
        params = {
            "location": f"{lat},{lng}",
            "radius": radius,
            "type": "tourist_attraction",
            "key": self.api_key
        }
        
        response = requests.get(endpoint, params=params)
        return response.json()
    
    def get_hotels(self, location, radius=5000):
        """Get hotels near a location."""
        endpoint = f"{self.BASE_URL}/place/nearbysearch/json"
        
        # First geocode the location to get coordinates
        geocode_result = self.geocode(location)
        if not geocode_result.get("results"):
            return {"error": "Could not geocode location"}
        
        lat = geocode_result["results"][0]["geometry"]["location"]["lat"]
        lng = geocode_result["results"][0]["geometry"]["location"]["lng"]
        
        params = {
            "location": f"{lat},{lng}",
            "radius": radius,
            "type": "lodging",
            "key": self.api_key
        }
        
        response = requests.get(endpoint, params=params)
        return response.json()