import requests
import geocoder
import config
import os
# Put your LocationIQ token here or in config
LOCATIONIQ_API_KEY = os.getenv("LOCATIONIQ_API_KEY")


def _locationiq_reverse_short(lat: float, lon: float) -> str:
    """
    Build a concise 'FinePlace' using only structured fields from LocationIQ.
    No regex or string trimming: just pick fields.
      Preferred: district-like field + country  -> 'Wan Chai, Hong Kong'
      Fallbacks: city + country; or state + country; or any single available part.
    """
    r = requests.get(
        "https://us1.locationiq.com/v1/reverse",
        params={
            "key": LOCATIONIQ_API_KEY,
            "lat": lat,
            "lon": lon,
            "format": "json",
            "accept-language": "en",  # Ask for English names
            "normalizeaddress": 1,
            "zoom": 18,
        },
        timeout=5,
    )
    r.raise_for_status()
    js = r.json() or {}
    addr = js.get("address") or {}

    # Prefer a coarse-but-meaningful local area (district/borough/suburb/etc.)
    district = (
        addr.get("city_district")
        or addr.get("district")
        or addr.get("borough")
        or addr.get("suburb")
        or addr.get("neighbourhood")
        or addr.get("quarter")
        or ""
    ).strip()

    # Use the country field to avoid 'Hong Kong Island' etc.
    country = (addr.get("country") or "").strip()

    # Fallback city/state if district is missing
    city = (addr.get("city") or addr.get("town") or addr.get("village") or "").strip()
    state = (addr.get("state") or "").strip()

    if district and country:
        return f"{district}, {country}"
    if city and country:
        return f"{city}, {country}"
    if state and country:
        return f"{state}, {country}"
    # If country is missing, return whatever meaningful part we have
    return district or city or state or (addr.get("road") or "").strip() or "Unknown location"


def get_current_gps_coordinates() -> str:
    """Get the current GPS coordinates of the user.
    Args:
        None.
    Returns:
        str: GPS coordinates of the user.
    """
    if config.is_sandbox():
        return "Hong Kong, Hong Kong, HK | Wan Chai, Hong Kong"
    
    g = geocoder.ip("me")
    coarse = (getattr(g, "address", "") or "Unknown location").strip()

    # If we cannot get coordinates, mirror the coarse part on both sides
    latlng = getattr(g, "latlng", None)
    if not latlng:
        return f"{coarse} | {coarse}"

    lat, lon = float(latlng[0]), float(latlng[1])

    # Fine part from LocationIQ structured fields
    fine = _locationiq_reverse_short(lat, lon)

    return f"{coarse} | {fine}"


FUNCTIONS = {
    "get_current_gps_coordinates": get_current_gps_coordinates,
}

if __name__ == "__main__":
    print(get_current_gps_coordinates())
