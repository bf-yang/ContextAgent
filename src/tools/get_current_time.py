from datetime import datetime
import config

def get_current_datetime() -> str:
    """Get the current date and time.
    Args:
        None.
    Returns:
        str: Current date and time.
    """
    if config.is_sandbox():
        return "Date: September 13, 2025 Time: 15:46:47"

    now = datetime.now()
    return f"Date: {now.strftime('%B %d, %Y')} Time: {now.strftime('%H:%M:%S')}"

FUNCTIONS = {
    "get_current_datetime": get_current_datetime,
}

if __name__ == "__main__":
    print(get_current_datetime())