from typing import Optional
import requests
import config
import os
GOOGLE_MAP_API_KEY = os.getenv("GOOGLE_MAP_API_KEY")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

def _point(address: str):
    if not AMAP_API_KEY:
        raise RuntimeError("AMAP_API_KEY not set")
    r = requests.get("https://restapi.amap.com/v3/place/text",
                     params={"keywords": address, "key": AMAP_API_KEY, "offset": 1, "page": 1},
                     timeout=10)
    r.raise_for_status()
    pois = (r.json().get("pois") or [])
    if pois and pois[0].get("location"):
        lng, lat = map(float, pois[0]["location"].split(","))
        return lng, lat
    r = requests.get("https://restapi.amap.com/v3/geocode/geo",
                     params={"address": address, "key": AMAP_API_KEY},
                     timeout=10)
    r.raise_for_status()
    geos = (r.json().get("geocodes") or [])
    if geos and geos[0].get("location"):
        lng, lat = map(float, geos[0]["location"].split(","))
        return lng, lat
    raise ValueError(f"Address not found: {address}")

def _fmt_duration(minutes: float, lang="en"):
    m = int(round(minutes))
    h, m = divmod(m, 60)
    if lang == "en":
        return f"{h} hr {m} min" if h else f"{m} min"
    else:
        return f"{h}小时{m}分钟" if h else f"{m}分钟"  # Chinese output for duration

def amap_route(origin: str, destination: str, lang: str = "en", units: str = "km") -> str:
    """
    lang: 'en' or 'zh'
    units: 'km' or 'mi'
    """
    o_lng, o_lat = _point(origin)
    d_lng, d_lat = _point(destination)

    r = requests.get("https://restapi.amap.com/v3/direction/driving",
                     params={"origin": f"{o_lng},{o_lat}",
                             "destination": f"{d_lng},{d_lat}",
                             "key": AMAP_API_KEY,
                             "extensions": "base"},
                     timeout=10)
    r.raise_for_status()
    paths = (r.json().get("route") or {}).get("paths") or []
    if not paths:
        raise ValueError("No route found")

    p = paths[0]
    km = int(p["distance"]) / 1000
    minutes = int(p["duration"]) / 60

    if units == "mi":
        dist_val = km * 0.621371
        dist_str = f"{dist_val:.2f} mi"
    else:
        dist_str = f"{km:.2f} km"

    dur_str = _fmt_duration(minutes, lang)

    if lang == "en":
        return f"{origin} → {destination}: {dist_str}, approx {dur_str}"
    else:
        return f"{origin} → {destination}：{dist_str}，约 {dur_str}"  # Chinese output for route

def google_map(current_location: str, destination: str) -> str:
    """
    Get the route and distance from the current location to the destination using Google Maps API.

    Args:
        current_location (str): The starting location.
        destination (str): The destination location.

    Returns:
        str: The route and distance information.
    """
    if destination is None or not str(destination).strip():
        return (
            f"You are at {current_location}. "
            "Do you want to get directions to a destination? "
            "Provide a destination to receive distance and ETA."
        )
    
    if config.is_sandbox():
        return f"The distance from {current_location} to {destination} is 2 km, approx 36 min."

    # # Google Map
    # gmaps = googlemaps.Client(GOOGLE_MAP_API_KEY)
    # current_location = get_current_gps_coordinates()
    # response = gmaps.distance_matrix(current_location, destination, mode = 'driving')
    # return response

    # AMAP
    response = amap_route(current_location, destination)
    return response

FUNCTIONS = {
    "google_map": google_map,
}


if __name__ == "__main__":
    # Example usage
    print(google_map("Wan Chai, Hong Kong", "Mong Kok"))