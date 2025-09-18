import os

# Recommended names
MODES = {"live", "sandbox"}
DEFAULT_MODE = "live"

# Priority: ENV -> default
MODE = os.getenv("APP_MODE", DEFAULT_MODE).lower()
if MODE not in MODES:
    MODE = DEFAULT_MODE

def set_mode(mode: str) -> None:
    """Set global mode at runtime (used by main after argparse)."""
    m = (mode or "").lower()
    if m not in MODES:
        raise ValueError(f"Unknown mode '{mode}', must be one of {sorted(MODES)}")
    global MODE
    MODE = m

def is_sandbox() -> bool:
    return MODE != "live"
