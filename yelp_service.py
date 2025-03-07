import os
import requests

class YelpService:
    BASE_URL = "https://api.yelp.com/v3"
    
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.getenv("YELP_API_KEY")
        
        if not api_key:
            raise ValueError("Yelp API key is required")
        
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def search_businesses(self, term=None, location=None, latitude=None, longitude=None, 
                          categories=None, price=None, open_now=None, limit=20):
        """
        Search for businesses on Yelp.
        
        Parameters:
        - term: Search term (e.g., "food", "restaurants")
        - location: Location name (e.g., "San Francisco, CA")
        - latitude, longitude: Coordinates for location-based search
        - categories: List of category aliases (e.g., ["restaurants", "bars"])
        - price: Price level, 1-4 as a comma-delimited string (e.g., "1,2,3")
        - open_now: Whether the business is open now
        - limit: Number of results to return (max 50)
        """
        endpoint = f"{self.BASE_URL}/businesses/search"
        
        params = {}
        if term:
            params["term"] = term
        if location:
            params["location"] = location
        if latitude and longitude:
            params["latitude"] = latitude
            params["longitude"] = longitude
        if categories:
            params["categories"] = ",".join(categories) if isinstance(categories, list) else categories
        if price:
            params["price"] = price
        if open_now is not None:
            params["open_now"] = open_now
        if limit:
            params["limit"] = limit
            
        response = requests.get(endpoint, headers=self.headers, params=params)
        return response.json()
    
    def get_business(self, business_id):
        """Get detailed information about a business."""
        endpoint = f"{self.BASE_URL}/businesses/{business_id}"
        response = requests.get(endpoint, headers=self.headers)
        return response.json()
    
    def get_reviews(self, business_id):
        """Get reviews for a business."""
        endpoint = f"{self.BASE_URL}/businesses/{business_id}/reviews"
        response = requests.get(endpoint, headers=self.headers)
        return response.json()
    
    def get_restaurants(self, location, cuisine=None, price=None, limit=5):
        """Get restaurants in a location with optional cuisine and price filters."""
        categories = cuisine if cuisine else "restaurants"
        return self.search_businesses(
            term="restaurants", 
            location=location,
            categories=categories,
            price=price,
            limit=limit
        )
    
    def get_activities(self, location, category=None, limit=5):
        """Get activities or attractions in a location."""
        # Default categories for activities if none specified
        if not category:
            category = "active,arts,tours"
        
        return self.search_businesses(
            term="activities",
            location=location,
            categories=category,
            limit=limit
        )