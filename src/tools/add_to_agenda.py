from typing import Any, Optional, Tuple
from gcsa.google_calendar import GoogleCalendar
from gcsa.event import Event
from datetime import datetime, timedelta, date
import dateparser, re
import config
import os

GOOGLE_CALENDAR_ACCOUNT = os.getenv("GOOGLE_CALENDAR_ACCOUNT")


# Weekday mapping for the tiny fallback parser
_WD = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2,
    "thu": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}

def _parse_time_natural(time_str: str, base: datetime | None = None) -> datetime:
    """
    Parse natural-language time to a datetime without LLM.
    1) Try dateparser with a cleaned string.
    2) Fallback: handle "this/next <weekday> at <time>" and "<weekday> [morning|afternoon|evening|night]".
    """
    base = base or datetime.now()
    s = (time_str or "").strip()
    if not s:
        raise ValueError("Time string is empty.")

    norm = re.sub(r"\b(this|coming)\b", "", s, flags=re.I).strip()

    dt = dateparser.parse(
        norm,
        languages=["en", "zh"],
        settings={
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": base,
            "RETURN_AS_TIMEZONE_AWARE": False,
        },
    )
    if dt:
        return dt

    def _weekday_to_target(qualifier: str, wd_key: str, hour: int, minute: int) -> datetime:
        target_idx = _WD[wd_key]
        today_idx = base.weekday()
        days_ahead = (target_idx - today_idx) % 7
        if qualifier == "next" or (days_ahead == 0 and (hour, minute) <= (base.hour, base.minute)):
            days_ahead += 7
        tgt = (base + timedelta(days=days_ahead)).date()
        return datetime(tgt.year, tgt.month, tgt.day, hour, minute)

    rx_time = re.compile(
        r"(?i)\b(this|next)?\s*"
        r"(mon|monday|tue|tues|tuesday|wed|wednesday|thu|thurs|thursday|fri|friday|sat|saturday|sun|sunday)"
        r"(?:\s*(?:at)?\s*)?"
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b"
    )
    m = rx_time.search(s)
    if m:
        qualifier = (m.group(1) or "").lower()
        wd_key    = m.group(2).lower()
        hour      = int(m.group(3))
        minute    = int(m.group(4) or 0)
        ampm      = (m.group(5) or "").lower()
        if ampm:
            if hour == 12: hour = 0
            if ampm == "pm": hour += 12
        return _weekday_to_target(qualifier, wd_key, hour, minute)

    rx_day_only = re.compile(
        r"(?i)\b(this|next)?\s*"
        r"(mon|monday|tue|tues|tuesday|wed|wednesday|thu|thurs|thursday|fri|friday|sat|saturday|sun|sunday)"
        r"(?:\s+(morning|afternoon|evening|night))?\b"
    )
    m2 = rx_day_only.search(s)
    if m2:
        qualifier = (m2.group(1) or "").lower()
        wd_key    = m2.group(2).lower()
        period    = (m2.group(3) or "").lower()
        default_hour = {"morning": 9, "afternoon": 15, "evening": 19, "night": 21}.get(period, 9)
        return _weekday_to_target(qualifier, wd_key, default_hour, 0)

    raise ValueError(f"Could not understand time: {time_str}")

def _coerce_event_datetime(time_input: Any, base: datetime | None = None) -> datetime:
    """
    Accept either:
      - a string like "next Saturday 3 pm" (parsed by _parse_time_natural), or
      - a dict {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"} from utils.parse_time_phrase.

    Returns a concrete start datetime. For dict input, we pick the midpoint date (inclusive)
    and assign a default time-of-day 09:00.
    """
    base = base or datetime.now()
    if isinstance(time_input, dict) and time_input.get("start_date"):
        try:
            s = datetime.fromisoformat(time_input["start_date"]).date()
            e = datetime.fromisoformat(time_input.get("end_date", time_input["start_date"])).date()
            if e < s:
                s, e = e, s
            mid = s + timedelta(days=((e - s).days // 2))
            return datetime(mid.year, mid.month, mid.day, 9, 0)
        except Exception as e:
            raise ValueError(f"Invalid date window: {e}")
    if isinstance(time_input, str):
        return _parse_time_natural(time_input, base=base)
    raise ValueError("Unsupported time format; expected string or {'start_date','end_date'} dict.")

def add_to_agenda(event: str, time: Any) -> str:
    """
    Add an event to the user's agenda.

    Args:
        event (str): The name of the event to add.
        time (str|dict): Natural-language time string or {'start_date','end_date'} window.

    Returns:
        str: Confirmation message or error string.
    """
    if config.is_sandbox():
        return "Event 'Group meeting' added for 2025-09-19 15:00."

    # Parse time (string or dict) into a precise datetime
    try:
        start_dt = _coerce_event_datetime(time, base=datetime.now())
    except Exception as e:
        return f"Error: {e}"

    # Create the event via GCSA
    try:
        calendar = GoogleCalendar(GOOGLE_CALENDAR_ACCOUNT)
        ev = Event(
            event,
            start=start_dt,
            minutes_before_popup_reminder=15,
            # end=start_dt + timedelta(hours=1),  # optional duration
        )
        calendar.add_event(ev)
    except Exception as e:
        return f"Error adding event: {e}"

    return f"Event '{event}' added for {start_dt:%Y-%m-%d %H:%M}."


FUNCTIONS = {
    "add_to_agenda": add_to_agenda,
}

if __name__ == "__main__":
    # print(add_to_agenda("Group meeting", "This Friday at 3 PM"))
    # print(add_to_agenda("Group meeting", "Next Monday at 10:30 AM"))
    print(add_to_agenda("Group meeting", "Next Monday"))