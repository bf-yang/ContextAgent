import json
from pathlib import Path
from datetime import datetime, timedelta
import config

DB_PATH = Path("data/tools/meetings_db.json")

def _load_db() -> dict:
    # Load the meeting database from file
    if DB_PATH.exists():
        try:
            return json.loads(DB_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_db(db: dict) -> None:
    # Save the meeting database to file
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")

def add_meeting(meeting_topic: str, start_time: str, meeting_location: str) -> str:
    """
    This API allows users to make a reservation for a meeting and store the meeting information (e.g., topic, time, location, attendees) in the database.

    Args:
        meeting_topic (str): The topic of the meeting.
        start_time (str): The start time of the meeting, in the pattern of %Y-%m-%d %H:%M:%S'.
        meeting_location (str): The location where the meeting to be held.

    Returns:
        status (str): success or failed.
    """
    if config.is_sandbox():
        return "success: 1."


    topic = (meeting_topic or "").strip()
    loc   = (meeting_location or "").strip()
    if not topic:
        return "Error: meeting_topic cannot be empty."
    if not loc:
        return "Error: meeting_location cannot be empty."

    # Validate and parse start time
    try:
        dt_start = datetime.strptime((start_time or "").strip(), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return "Error: start_time must match '%Y-%m-%d %H:%M:%S'."

    # Default duration is 1 hour
    dt_end = dt_start + timedelta(hours=1)

    # Load database and generate auto-increment ID
    db = _load_db()
    next_id = str(max([int(k) for k in db.keys()] + [0]) + 1)

    # Store meeting record
    record = {
        "meeting_topic": topic,
        "start_time": dt_start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": dt_end.strftime("%Y-%m-%d %H:%M:%S"),
        "location": loc,
    }
    db[next_id] = record
    try:
        _save_db(db)
    except Exception as e:
        return f"Error: failed to save database ({e})"

    return f"success: {next_id}"

FUNCTIONS = {
    "add_meeting": add_meeting,
}

if __name__ == "__main__":
    print(add_meeting("Project Sync", "2025-09-12 10:00:00", "Room 301"))