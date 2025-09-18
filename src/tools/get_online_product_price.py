import requests
import config
import os
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def get_online_product_price(product_name: str) -> str:
    """
    Get the price of a product from an online store such as Amazon or eBay.

    Args:
        product_name (str): The name of the product to search for.

    Returns:
        str: The price of the product as a string.
    """

    if config.is_sandbox():
        return "Apple Iphone 15 Pro Max - 256gb - Black Titanium - T-mobile - Fair — $519.00 "
   
    if not SERPAPI_KEY:
            return "SERPAPI_KEY not set"
    try:
        resp = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_shopping",  # Use Shopping engine
                "q": product_name,
                "hl": "en", "gl": "us",       # Region/language, can be adjusted as needed
                "api_key": SERPAPI_KEY,
            },
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("shopping_results", []) or []
        if not items:
            return f"No price found for {product_name}"

        # Select the lowest extracted_price; if not available, set to a very large value
        best = min(items, key=lambda x: x.get("extracted_price") or 1e18)
        title = best.get("title") or product_name
        # `price` is a string with currency symbol, `extracted_price` is a pure number
        price_str = best.get("price") or f"${best.get('extracted_price')}"
        link = best.get("link") or ""
        return f"{title} — {price_str} {('(' + link + ')') if link else ''}"
    except Exception as e:
        return f"Lookup failed: {e}"
    
FUNCTIONS = {
    "get_online_product_price": get_online_product_price,
}
    
if __name__ == "__main__":
    product = "iphone 15 Pro Max"
    price_info = get_online_product_price(product)
    print(price_info)