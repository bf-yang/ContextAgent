import re, threading, time
from datetime import datetime, timedelta
import config


# ---------- Helpers (duration parsing & formatting) ----------

def _parse_duration_to_seconds(s: str) -> float:
    """
    Parse a duration string into seconds.
    Supported examples: '30s', '5m', '1h30m', '00:45:00', '10sec', '25min', '2h', or a pure number (seconds).
    NOTE: Bare 'HH:MM(:SS)' without am/pm/date is treated as a DURATION (backward-compatible).
    """
    if not s or not str(s).strip():
        raise ValueError("duration is empty")
    s = str(s).strip().lower()

    # 'HH:MM' or 'HH:MM:SS' -> interpret as DURATION
    if ":" in s and all(part.isdigit() for part in s.split(":")):
        parts = [int(p) for p in s.split(":")]
        if len(parts) == 2:
            h, m = parts
            return float(h * 3600 + m * 60)
        if len(parts) == 3:
            h, m, sec = parts
            return float(h * 3600 + m * 60 + sec)
        raise ValueError(f"invalid HH:MM(:SS) duration: {s}")

    total = 0.0
    # English and Chinese unit tokens for robustness
    for num, unit in re.findall(r'(\d+(?:\.\d*)?)\s*(d|day|天|h|hr|hour|小时|m|min|minute|分钟|分|s|sec|second|秒)', s):
        val = float(num)
        if unit in ("d", "day", "天"):
            total += val * 86400
        elif unit in ("h", "hr", "hour", "小时"):
            total += val * 3600
        elif unit in ("m", "min", "minute", "分钟", "分"):
            total += val * 60
        elif unit in ("s", "sec", "second", "秒"):
            total += val
    if total > 0:
        return total

    # Pure number -> seconds
    return float(s)


def _humanize(seconds: float) -> str:
    """Format seconds into a compact human-readable string like '1h 5m'."""
    seconds = int(round(seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if d: parts.append(f"{d}d")
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    if s or not parts: parts.append(f"{s}s")
    return " ".join(parts)


# ---------- Absolute time parsing ----------

_AMPM_OR_DATE = re.compile(
    r'(am|pm|\b(today|tomorrow|今天|明天)\b|\d{4}-\d{1,2}-\d{1,2})',
    re.IGNORECASE
)

def _looks_like_absolute(spec: str) -> bool:
    """
    Heuristic: treat as ABSOLUTE time if the string contains:
      - 'am' or 'pm', or
      - 'today' / 'tomorrow' (English) or their Chinese equivalents, or
      - an explicit date 'YYYY-MM-DD'.
    Otherwise, keep DURATION semantics (including bare 'HH:MM(:SS)').
    """
    return bool(_AMPM_OR_DATE.search(spec))


def _parse_alarm_to_datetime(when: str) -> datetime:
    """
    Parse an absolute-time expression into a local datetime.
    Examples: '7:00 am tomorrow', '19:30 today', '2025-09-20 07:00', '7am', '7:30pm'.
    Rules:
      - If no explicit date is provided, assume today; if that time has already passed, shift to tomorrow.
      - Supports both 12-hour (am/pm) and 24-hour formats.
      - Also recognizes Chinese tokens for 'today'/'tomorrow', though comments are in English.
    """
    if not when or not str(when).strip():
        raise ValueError("when is empty")
    s = str(when).strip().lower()

    # Optional leading 'at '
    if s.startswith("at "):
        s = s[3:].strip()

    now = datetime.now()

    # Relative-day hints
    is_today = any(tok in s for tok in ("today", "今天"))
    is_tomorrow = any(tok in s for tok in ("tomorrow", "明天"))

    # Explicit date: YYYY-MM-DD (time may follow)
    m_date = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', s)
    ymd = tuple(map(int, m_date.groups())) if m_date else None

    # Time: H[:MM[:SS]] with optional am/pm
    m_time = re.search(r'(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?\s*(am|pm)?', s)
    if not m_time:
        raise ValueError(f"cannot parse time from: {when}")
    hh = int(m_time.group(1))
    mm = int(m_time.group(2) or 0)
    ss = int(m_time.group(3) or 0)
    ampm = (m_time.group(4) or "").lower()

    # 12-hour normalization
    if ampm:
        if hh == 12:
            hh = 0
        if ampm == "pm":
            hh += 12

    # Construct the target datetime
    if ymd:
        y, m, d = ymd
        target = datetime(y, m, d, hh, mm, ss)
    else:
        base = now + timedelta(days=1) if is_tomorrow else now
        target = datetime(base.year, base.month, base.day, hh, mm, ss)
        # If 'today/tomorrow' not specified and the time for today has passed, roll to tomorrow
        if not (is_today or is_tomorrow) and target <= now:
            target += timedelta(days=1)

    return target


# ---------- Unified API: one function for duration & absolute ----------

def set_timer(duration: str, message: str = "⏰ Timer!", background: bool = True) -> str:
    """
    Set a timer for a specific duration.

    Args:
        duration (str): The duration of the timer.

    Returns:
        str: The timer set confirmation.
    """
    if config.is_sandbox():
        return "⏰ Timer up! (at 2025-09-11 22:50:21)"

    try:
        if _looks_like_absolute(duration):
            # Absolute-time alarm
            target = _parse_alarm_to_datetime(duration)
            secs = (target - datetime.now()).total_seconds()
            if secs <= 0:
                return "Error: alarm time must be in the future"

            def _ding():
                print(f"{message} (at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

            if background:
                t = threading.Timer(secs, _ding)
                # Not daemon, so the main process will wait for the timer unless explicitly terminated
                t.daemon = False
                t.start()
                return f"Alarm set for {target.strftime('%Y-%m-%d %H:%M:%S')} (in {_humanize(secs)})"
            else:
                time.sleep(secs)
                _ding()
                return f"Alarm triggered after {_humanize(secs)} at {target.strftime('%Y-%m-%d %H:%M:%S')}"
        else:
            # Duration countdown
            secs = _parse_duration_to_seconds(duration)
            if secs <= 0:
                return "Error: duration must be > 0 seconds"

            eta = datetime.now() + timedelta(seconds=secs)

            def _ding():
                print(f"{message} (at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

            if background:
                t = threading.Timer(secs, _ding)
                t.daemon = False
                t.start()
                return f"Timer set for {_humanize(secs)} (ETA {eta.strftime('%Y-%m-%d %H:%M:%S')})"
            else:
                time.sleep(secs)
                _ding()
                return f"Timer finished after {_humanize(secs)}"
    except Exception as e:
        return f"Error: {e}"


FUNCTIONS = {
    "set_timer": set_timer,
}

if __name__ == "__main__":
    # Duration example
    # print(set_timer("10s"))
    # Absolute time examples
    print(set_timer("7:00 am tomorrow"))
    # print(set_timer("19:30 today"))
    # print(set_timer("2025-09-20 07:00"))
