import pandas as pd
import config

def get_health_data():
    """
    Get health data from the user's smart device.

    Args:
        None.

    Returns:
        str: The health data as a string.
    """
    # See: https://python-fitbit.readthedocs.io/en/latest/
    
    if config.is_sandbox():
        return (
            "Health summary (from CSV)\n"
            "Range: 2025-09-01 → 2025-09-05\n"
            "Total: 25,000 steps, 150 min, 2,500 kcal\n"
            "Intensity minutes: Sedentary 300, Light 100, Fair 30, Very 20\n"
            "Top activities: Walking (90 min), Running (40 min), Cycling (20 min)\n\n"
            "Recent 5 entries:\n"
            "- 2025-09-05 18:30 • Walking • 30 min • 4,000 steps • 400 kcal\n"
            "- 2025-09-05 07:00 • Running • 20 min • 3,000 steps • 300 kcal\n"
            "- 2025-09-04 19:00 • Cycling • 20 min • 2,500 steps • 250 kcal\n"
            "- 2025-09-03 18:00 • Walking • 25 min • 3,500 steps • 350 kcal\n"
            "- 2025-09-02 07:30 • Running • 20 min • 3,000 steps • 300 kcal"
        )
    
    data_path = "data/tools/test_fitbit.csv"
    recent = 5

    df = pd.read_csv(data_path)
    required = [
        "Activity Name", "Start Time", "Duration (min)", "Steps", "Calories",
        "Sedentary (min)", "Lightly (min)", "Fairly (min)", "Very (min)"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return f"Missing columns in CSV: {missing}"

    # parse time & sort
    df["Start Time"] = pd.to_datetime(df["Start Time"], errors="coerce")
    df = df.sort_values("Start Time")

    # totals
    total_steps = int(pd.to_numeric(df["Steps"], errors="coerce").sum(skipna=True))
    total_cal = int(pd.to_numeric(df["Calories"], errors="coerce").sum(skipna=True))
    total_min = float(pd.to_numeric(df["Duration (min)"], errors="coerce").sum(skipna=True))

    sed_min = int(pd.to_numeric(df["Sedentary (min)"], errors="coerce").sum(skipna=True))
    light_min = int(pd.to_numeric(df["Lightly (min)"], errors="coerce").sum(skipna=True))
    fair_min = int(pd.to_numeric(df["Fairly (min)"], errors="coerce").sum(skipna=True))
    very_min = int(pd.to_numeric(df["Very (min)"], errors="coerce").sum(skipna=True))

    # date span
    start_dt = df["Start Time"].min()
    end_dt = df["Start Time"].max()
    span = f"{start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d}" if pd.notna(start_dt) and pd.notna(end_dt) else "N/A"

    # top activities (by duration)
    top_acts = (
        df.groupby("Activity Name")["Duration (min)"]
          .sum(min_count=1)
          .sort_values(ascending=False)
          .head(3)
    )
    top_acts_str = ", ".join(f"{name} ({mins:.0f} min)" for name, mins in top_acts.items()) or "N/A"

    # recent N entries
    recent_rows = (
        df.sort_values("Start Time", ascending=False)
          .head(recent)[["Start Time", "Activity Name", "Duration (min)", "Steps", "Calories"]]
    )

    recent_lines = []
    for _, r in recent_rows.iterrows():
        ts = r["Start Time"]
        recent_lines.append(
            f"- {ts:%Y-%m-%d %H:%M} • {r['Activity Name']} • {r['Duration (min)']:.0f} min • "
            f"{int(r['Steps'])} steps • {int(r['Calories'])} kcal"
        )
    recent_str = "\n".join(recent_lines) if recent_lines else "(no recent entries)"

    return (
        "Health summary (from CSV)\n"
        f"Range: {span}\n"
        f"Total: {total_steps:,} steps, {total_min:.0f} min, {total_cal:,} kcal\n"
        f"Intensity minutes: Sedentary {sed_min}, Light {light_min}, Fair {fair_min}, Very {very_min}\n"
        f"Top activities: {top_acts_str}\n\n"
        f"Recent {len(recent_lines)} entries:\n{recent_str}"
    )

FUNCTIONS = {
    "get_health_data": get_health_data,
}

if __name__ == "__main__":
    print(get_health_data())
