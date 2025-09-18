from typing import Optional, Iterable
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import config

CANDIDATES: list[str] = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]  # Common 5 stocks

def query_stock(stock_code: Optional[str] = None,
                date: Optional[str] = None,
                field: str = "Close") -> str:
    """
    This API queries the stock price of a given stock code and date.

    Args:
        stock_code (str): The stock code of the given stock.
        date (str): The date of the stock price. Format: %Y-%m-%d.

    Returns:
        stock_price (str): The stock price of the given stock.
    """
    if config.is_sandbox():
        return "AAPL 2023-09-11 Close: 174.07."

    # Parse date
    if date is None or not str(date).strip() or str(date).strip().lower() in ("today", "now"):
        dt = datetime.today().date()
        date_str = dt.strftime("%Y-%m-%d")
    else:
        try:
            dt = datetime.strptime(date.strip(), "%Y-%m-%d").date()
            date_str = date.strip()
        except Exception:
            return f"Error: invalid date '{date}', expect YYYY-MM-DD."

    # List of codes to query
    codes: Iterable[str] = [stock_code] if (stock_code or "").strip() else CANDIDATES

    lines = []
    for raw in codes:
        code = (raw or "").upper().strip()
        if not code:
            lines.append("Error: stock_code cannot be empty.")
            continue

        # Fetch a small window to handle non-trading days
        start = (dt - timedelta(days=10)).strftime("%Y-%m-%d")
        end   = (dt + timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance: end is exclusive
        try:
            hist = yf.Ticker(code).history(start=start, end=end, interval="1d", auto_adjust=False)
        except Exception as e:
            lines.append(f"{code} {date_str} {field}: Error fetching data: {e}")
            continue

        if hist is None or hist.empty:
            lines.append(f"{code} {date_str} {field}: No data around this date.")
            continue

        # Take the last record <= target date (automatically rollback for weekends/market closure)
        hist = hist.reset_index()
        hist["DateOnly"] = pd.to_datetime(hist["Date"]).dt.date
        sub = hist[hist["DateOnly"] <= dt]
        if sub.empty:
            lines.append(f"{code} {date_str} {field}: No trading data on/before this date.")
            continue

        row = sub.iloc[-1]
        used_date = row["DateOnly"].strftime("%Y-%m-%d")
        if field not in row:
            lines.append(f"{code} {date_str} {field}: Field not available.")
            continue

        val = float(row[field])
        note = "" if used_date == date_str else f" (used previous trading day {used_date})"
        lines.append(f"{code} {date_str} {field}: {val:.2f}{note}")

    return "\n".join(lines)

FUNCTIONS = {
    "query_stock": query_stock,
}

if __name__ == "__main__":
    # print(query_stock("AAPL", "2025-9-11"))
    # print(query_stock("AAPL"))  
    print(query_stock())