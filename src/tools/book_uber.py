from typing import Optional
import requests
import config
import os
AMAP_API_KEY = os.getenv("AMAP_API_KEY")
USER_AGENT = "contextagent/1.0 (contextagent@gmail.com)" 

def _geocode_amap(q: str):
    """
    AMAP fallback geocoder (no map usage, just geocoding).
    Try place/text first (POI), then geocode/geo.
    Returns (lat, lon, name) on success, or None on failure.
    """
    if not AMAP_API_KEY:
        return None

    # 1) POI search
    try:
        r = requests.get(
            "https://restapi.amap.com/v3/place/text",
            params={"keywords": q, "key": AMAP_API_KEY, "offset": 1, "page": 1},
            timeout=10,
        )
        r.raise_for_status()
        pois = r.json().get("pois") or []
        if pois and pois[0].get("location"):
            lng, lat = map(float, pois[0]["location"].split(","))  # AMAP returns "lng,lat"
            name = pois[0].get("name") or q
            return lat, lng, name  # <-- fixed
    except Exception:
        pass

    # 2) Address geocoding
    try:
        r = requests.get(
            "https://restapi.amap.com/v3/geocode/geo",
            params={"address": q, "key": AMAP_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        geos = r.json().get("geocodes") or []
        if geos and geos[0].get("location"):
            lng, lat = map(float, geos[0]["location"].split(","))  # "lng,lat"
            name = geos[0].get("formatted_address") or q
            return lat, lng, name  # <-- fixed
    except Exception:
        pass

    return None


def _geocode(q: str):
    """
    Primary geocoder: Nominatim (no key).
    Fallback: AMAP (uses your AMAP_API_KEY) when Nominatim finds nothing.
    Returns (lat, lon, display_name).
    """
    # --- Try Nominatim first ---
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": q,
                "format": "json",
                "limit": 1,
                "accept-language": "en",  # English names
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            name = data[0].get("display_name", q)
            return lat, lon, name
    except Exception:
        # swallow and try fallback
        pass

    # --- Fallback: AMAP ---
    amap_res = _geocode_amap(q)
    if amap_res:
        return amap_res  # (lat, lon, name)

    # Nothing found
    raise ValueError(f"Address not found: {q}")


def _route_osrm(lat_o, lon_o, lat_d, lon_d):
    """OSRM: get shortest-route distance (meters) and duration (seconds)."""
    url = f"https://router.project-osrm.org/route/v1/driving/{lon_o},{lat_o};{lon_d},{lat_d}"
    r = requests.get(url, params={"overview": "false"}, timeout=10)
    r.raise_for_status()
    routes = r.json().get("routes") or []
    if not routes:
        raise ValueError("No route found")
    dist_m = routes[0]["distance"]
    dur_s = routes[0]["duration"]
    return dist_m, dur_s


def _format_duration(minutes: float):
    m = int(round(minutes))
    h, m = divmod(m, 60)
    return f"{h} hr {m} min" if h else f"{m} min"


def estimate_ride(origin: str, destination: str, *, rates=None, currency="USD", units="km") -> str:
    """
    Estimate ride time and fare (keyless stack).
    rates: fare dict, e.g. {"base":2.5, "per_km":1.3, "per_min":0.3, "booking":2.0}
    units: "km" or "mi"
    """
    if rates is None:
        rates = {"base": 2.5, "per_km": 1.3, "per_min": 0.3, "booking": 2.0}

    # 1) Geocoding
    o_lat, o_lon, _ = _geocode(origin)
    d_lat, d_lon, _ = _geocode(destination)

    # 2) Routing
    dist_m, dur_s = _route_osrm(o_lat, o_lon, d_lat, d_lon)
    km = dist_m / 1000.0
    minutes = dur_s / 60.0

    # 3) Fare estimate
    fare = rates["base"] + rates["booking"] + km * rates["per_km"] + minutes * rates["per_min"]

    # Units
    if units == "mi":
        dist_val = km * 0.621371
        dist_str = f"{dist_val:.2f} mi"
    else:
        dist_str = f"{km:.2f} km"

    return (
        f"Estimate for {origin} → {destination}: "
        f"{dist_str}, ~{_format_duration(minutes)}, "
        f"~{currency} {fare:.2f} "
        f"(base {rates['base']} + booking {rates['booking']} + "
        f"{km:.1f} km × {rates['per_km']}/km + {int(minutes)} min × {rates['per_min']}/min)"
    )


def book_uber(current_location: str, destination: Optional[str] = None) -> str:
    """
    Prompt the user with the estimated time and cost of the Uber ride, and ask whether they would like to proceed with the booking.

    Args:
        current_location (str): The starting location.
        destination (str): The destination location.

    Returns:
        str: The Uber ride booking confirmation.
    """
    if destination is None or not str(destination).strip():
        return (
            f"You are at {current_location}. "
            "Do you want to request an Uber? If so, please provide the destination."
        )

    if config.is_sandbox():
        return (
            f"Estimate for {current_location} to {destination}: "
            "15.86 km, ~20 min, ~USD 31.14 "
            "(base 2.5 + booking 2.0 + 15.9 km × 1.3/km + 20 min × 0.3/min)."
        )

    return estimate_ride(current_location, destination)


FUNCTIONS = {
    "book_uber": book_uber,
}

if __name__ == "__main__":
    print(book_uber("Wan Chai, Hong Kong", "City University of Hong Kong"))