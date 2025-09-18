import requests
from typing import Any, Optional, Tuple, List
from datetime import datetime, date, timedelta
import config

USER_AGENT = "weather-tool/0.1 (contextagent@gmail.com)"  # arbitrary UA


def _parse_date(s: str) -> date:
    return datetime.fromisoformat(s).date()

def _center_date(time_arg: Any) -> Optional[date]:
    """
    Determine the target 'center' date:
    - If time_arg is a string like "this weekend" or "next_weekend": map to Saturday of this/next week.
    - If time_arg is {"start_date","end_date"}: use midpoint (inclusive).
    - If only start_date exists: use that day.
    - Otherwise: None (no explicit target).
    """
    # New: accept "this weekend" / "next_weekend" style strings (case-insensitive; spaces/underscores both work)
    if isinstance(time_arg, str):
        s = time_arg.strip().lower().replace("_", " ")
        if s in {"this weekend", "weekend"}:
            today = date.today()
            # Monday=0 ... Saturday=5, Sunday=6
            days_until_sat = (5 - today.weekday()) % 7
            return today + timedelta(days=days_until_sat)
        if s in {"next weekend", "next wknd"}:
            today = date.today()
            days_until_sat = (5 - today.weekday()) % 7
            return today + timedelta(days=days_until_sat + 7)

    if isinstance(time_arg, dict) and time_arg.get("start_date"):
        s = _parse_date(time_arg["start_date"])
        e = _parse_date(time_arg.get("end_date", time_arg["start_date"]))
        if e < s:
            s, e = e, s
        # midpoint (floor)
        days = (e - s).days
        return s + timedelta(days=days // 2)
    return None


def get_city_weather(city: str, time: Any) -> str:
    """Get daily weather, default 3 days; if a target date is provided, return the 3-day window around it."""
    if config.is_sandbox():
        # keep sandbox behavior as-is
        return (
            " --> <DailyForecast date=2025-09-10 temperature=28°C>"
            " --> <DailyForecast date=2025-09-11 temperature=29°C>"
            " --> <DailyForecast date=2025-09-12 temperature=27°C>"
        )

    city = (city or "").strip()
    if not city:
        return "Error: city cannot be empty."

    # Compute center day (optional). If present, we'll fetch [center-1, center, center+1].
    center = _center_date(time)

    try:
        # 1) Geocoding
        g = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        g.raise_for_status()
        gj = g.json()
        if not gj.get("results"):
            return f"Error: city not found: {city}"
        lat = gj["results"][0]["latitude"]
        lon = gj["results"][0]["longitude"]

        # 2) Forecast params (Celsius)
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max",
            "temperature_unit": "celsius",
            "timezone": "auto",
        }

        # If we have a center day, ask the API for exactly the 3-day window.
        if center:
            start_api = (center - timedelta(days=1)).isoformat()
            end_api = (center + timedelta(days=1)).isoformat()
            params["start_date"] = start_api
            params["end_date"] = end_api

        f = requests.get("https://api.open-meteo.com/v1/forecast", params=params,
                         headers={"User-Agent": USER_AGENT}, timeout=10)
        f.raise_for_status()
        fj = f.json()
        daily = fj.get("daily") or {}
        dates: List[str] = daily.get("time") or []
        temps: List[Optional[float]] = daily.get("temperature_2m_max") or []
        rows = list(zip(dates, temps))
        if not rows:
            return "Error: no forecast data."

        # 3) Local guard filter to exactly the 3-day window around center (handles TZ quirks or API leniency)
        if center:
            wanted = {
                (center - timedelta(days=1)).isoformat(),
                center.isoformat(),
                (center + timedelta(days=1)).isoformat(),
            }
            rows = [(d, t) for d, t in rows if d in wanted]
            if not rows:
                return (f"No forecast available for "
                        f"{(center - timedelta(days=1)).isoformat()} to {(center + timedelta(days=1)).isoformat()}.")

        # 4) No center provided -> keep prior default: first 3 days
        if not center:
            rows = rows[:3]

        # 5) Render
        parts = []
        for d, t in rows:
            temp_txt = "?" if t is None else str(int(round(t)))
            parts.append(f" --> <DailyForecast date={d} temperature={temp_txt}°C>")
        return "".join(parts)

    except requests.RequestException as e:
        return f"Network error: {e}"
    except Exception as e:
        return f"Error: {e}"



FUNCTIONS = {
    "get_city_weather": get_city_weather,
}

if __name__ == "__main__":
    # print(get_city_weather("Hong Kong", {"start_date": "2025-09-27", "end_date": "2025-09-27"}))
    # Examples:
    print(get_city_weather("Hong Kong", "this weekend"))
    # print(get_city_weather("Hong Kong", "next_weekend"))
