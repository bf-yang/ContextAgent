from gcsa.google_calendar import GoogleCalendar
import config
from datetime import datetime, timedelta
import os

GOOGLE_CALENDAR_ACCOUNT = os.getenv("GOOGLE_CALENDAR_ACCOUNT")

def check_agenda_time_conflict() -> str:
    """
    Check user's agenda and return all events.

    Returns:
        str: All events in user's agenda.
    """
    if config.is_sandbox():
        return ["Event: Group Meeting, Start: 2025-09-10 15:00:00, End: 2025-09-10 16:00:00", "Event: Group Meeting, Start: 2025-09-11 15:00:00, End: 2025-09-11 16:00:00", "Event: Visa Appointment, Start: 2025-09-12 11:30:00, End: 2025-09-12 12:30:00"]

    calendar = GoogleCalendar(GOOGLE_CALENDAR_ACCOUNT)
    events_summary = []
    for event in calendar:
        event_start = event.start
        event_end = event.end if event.end else event.start + timedelta(hours=1)  # Assume 1-hour duration if no end time

        # Format event times as strings without timezone information
        event_start_str = event_start.strftime('%Y-%m-%d %H:%M:%S')
        event_end_str = event_end.strftime('%Y-%m-%d %H:%M:%S')

        event_summary = f"Event: {event.summary}, Start: {event_start_str}, End: {event_end_str}"
        events_summary.append(event_summary)

    return events_summary

FUNCTIONS = {
    "check_agenda_time_conflict": check_agenda_time_conflict,   
}

if __name__ == "__main__":
    agenda_summary = check_agenda_time_conflict()
    print(agenda_summary)