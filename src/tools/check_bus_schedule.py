import math, requests, datetime
from zoneinfo import ZoneInfo
import config

UA = "hk-bus-check/0.3 (you@example.com)"
NOMINATIM = "https://nominatim.openstreetmap.org/search"

# KMB/LWB
KMB_BASE     = "https://data.etabus.gov.hk/v1/transport/kmb"
KMB_STOPS    = f"{KMB_BASE}/stop"           # All stops
KMB_STOP_ETA = f"{KMB_BASE}/stop-eta"       # /{stop_id}

# Citybus
CITYBUS_BASE  = "https://rt.data.gov.hk/v2/transport/citybus"
CITYBUS_STOPS = f"{CITYBUS_BASE}/stop"      # All stops
CITYBUS_ETA   = f"{CITYBUS_BASE}/eta/CTB"   # /{stop_id}/{route} (route required)

HK_TZ = ZoneInfo("Asia/Hong_Kong")

def _geocode(q: str):
    r = requests.get(NOMINATIM, params={"q": q, "format": "json", "limit": 1},
                     headers={"User-Agent": UA}, timeout=12)
    r.raise_for_status()
    arr = r.json()
    if not arr:
        raise ValueError(f"Address not found: {q}")
    return float(arr[0]["lat"]), float(arr[0]["lon"]), arr[0].get("display_name", q)

def _dist_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _nearest(stops, lat, lon, radius_m=300, topn=5):
    bag = []
    for s in stops:
        try:
            d = _dist_m(lat, lon, float(s["lat"]), float(s.get("long") or s.get("lng")))
            if d <= radius_m:
                bag.append((d, s))
        except:
            pass
    bag.sort(key=lambda x: x[0])
    return bag[:topn]

def _load_json(url):
    r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    return r.json().get("data", [])

def _kmb_eta(stop_id: str):
    r = requests.get(f"{KMB_STOP_ETA}/{stop_id}", headers={"User-Agent": UA}, timeout=12)
    r.raise_for_status()
    return r.json().get("data", [])

def _ctb_eta(stop_id: str, route: str):
    r = requests.get(f"{CITYBUS_ETA}/{stop_id}/{route}", headers={"User-Agent": UA}, timeout=12)
    r.raise_for_status()
    return r.json().get("data", [])

def _fmt_eta(items, limit=3, route_filter=None):
    """
    Only show bus arrivals with ETA time; records without ETA (sometimes the API returns empty eta) will be skipped.
    If all records have no ETA, return an empty string to trigger a friendly message upstream.
    """
    out = []
    filt = (route_filter or "").upper().strip()
    for it in items:
        rt = (it.get("route") or "").upper()
        if filt and rt != filt:
            continue
        eta = it.get("eta") or it.get("eta_time")
        if not eta:  # Skip if no ETA
            continue
        dest = it.get("dest_en") or it.get("dest_tc") or ""
        rmk  = it.get("rmk_en") or it.get("rmk_tc") or ""
        out.append(f"{rt} → {dest}: {eta} {('('+rmk+')') if rmk else ''}".strip())
        if len(out) >= limit:
            break
    return "\n   ".join(out)

def _no_service_hint():
    now = datetime.datetime.now(HK_TZ)
    if now.hour >= 21:
        return "No upcoming buses. Service may have ended for today."
    return "No upcoming buses in the next ~30–60 min."

def check_bus_schedule(bus_stop: str,
                       provider: str = "auto",
                       route: str | None = None,
                       radius_m: int = 1000,
                       lang: str = "en") -> str:
    """
    Check the bus schedule for a specific bus stop.

    Args:
        bus_stop (str): The name of the bus stop.

    Returns:
        str: The bus schedule information.
    """
    if config.is_sandbox():
        return "Nearby stops for 'University, Vincent Drive, Birmingham Research Park, Bournbrook, Metchley, Birmingham, West Midlands, England, B15 2SG, United Kingdom' (≤1000 m):- KMB: No stops within radius."

    q = (bus_stop or "").strip()
    if not q:
        return "Error: please provide an address/landmark/stop id."

    # If it looks like a KMB stop number (digits), try KMB directly first
    if q.isdigit() and provider in ("auto", "kmb"):
        try:
            txt = _fmt_eta(_kmb_eta(q), route_filter=route)
            return f"KMB stop #{q}:\n   {txt if txt else _no_service_hint()}"
        except:
            pass  # If failed, go to address flow

    # Address to coordinates
    try:
        lat, lon, disp = _geocode(q)
    except Exception as e:
        return f"Geocoding failed: {e}"

    lines = [f"Nearby stops for '{disp}' (≤{radius_m} m):"]

    # KMB: Prefer stops with the specified route nearby; otherwise, show the nearest stop
    if provider in ("auto", "kmb"):
        try:
            near = _nearest(_load_json(KMB_STOPS), lat, lon, radius_m, topn=5)
            shown = False
            if near:
                if route:
                    for d, s in near:
                        txt = _fmt_eta(_kmb_eta(s["stop"]), route_filter=route)
                        if txt:
                            lines += [f"- KMB {s.get('name_en','')}/{s.get('name_tc','')} (#{s['stop']}, {int(d)} m)",
                                      "   " + txt]
                            shown = True
                            break
                if not shown:
                    d, s = near[0]
                    txt = _fmt_eta(_kmb_eta(s["stop"]))
                    lines += [f"- KMB {s.get('name_en','')}/{s.get('name_tc','')} (#{s['stop']}, {int(d)} m)",
                              "   " + (txt if txt else _no_service_hint())]
            else:
                lines.append("- KMB: No stops within radius.")
        except Exception:
            # Silent: If KMB API fails, do not show
            pass

    # Citybus: Route required; ignore errors silently
    if provider in ("auto", "citybus"):
        try:
            near = _nearest(_load_json(CITYBUS_STOPS), lat, lon, radius_m, topn=5)
            if near:
                s = near[0][1]
                if route:
                    txt = _fmt_eta(_ctb_eta(s["stop"], route.upper()))
                    if txt:
                        lines += [f"- Citybus {s.get('name_en','')}/{s.get('name_tc','')} (#{s['stop']}) route {route.upper()}",
                                  "   " + txt]
                    else:
                        lines += [f"- Citybus {s.get('name_en','')}/{s.get('name_tc','')} (#{s['stop']}) route {route.upper()}",
                                  "   " + _no_service_hint()]
                else:
                    # Do not force prompt, keep concise; uncomment next line if prompt is needed
                    # lines.append(f"- Citybus {s.get('name_en','')}/{s.get('name_tc','')} (#{s['stop']}) — ETA requires a route")
                    pass
        except Exception:
            pass  # Silent

    return "\n".join(lines)


FUNCTIONS = {
    "check_bus_schedule": check_bus_schedule,   
}

if __name__ == "__main__":
    print(check_bus_schedule("University Station, Sha Tin", route="87K"))
    # print(check_bus_schedule("University Station"))