"""
Polymarket daily high temperature market tracker — multi-city.

Supports: NYC (LaGuardia), Miami, Atlanta, Chicago, Seattle, Dallas.

Each city's data is stored under its key in the cache.  Every sync run
fetches up to 6 days of markets per city, computes the market-implied
temperature (probability-weighted bracket midpoints), and records the NWS
hourly forecast high for that city.  After a date passes the actual observed
daily high is backfilled from IEM ASOS.
"""

import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DATA_DIR = Path("/Volumes/T7/Weather_Models/data")
POLYMARKET_CACHE_PATH = DATA_DIR / "polymarket_cache.json"

GAMMA_API = "https://gamma-api.polymarket.com"

NWS_API = "https://api.weather.gov"
NWS_HEADERS = {
    "User-Agent": "(Weather Models App, contact@example.com)",
    "Accept": "application/geo+json",
}

# ---------------------------------------------------------------------------
# City configuration
# ---------------------------------------------------------------------------

CITIES = [
    {
        "key":          "nyc",
        "display":      "NYC",
        "slug_city":    "nyc",
        "nws_lat":      40.7769,
        "nws_lon":      -73.8740,
        "iem_station":  "LGA",       # IEM uses 3-letter codes (no leading K)
        "iem_network":  "NY_ASOS",
    },
    {
        "key":          "miami",
        "display":      "Miami",
        "slug_city":    "miami",
        "nws_lat":      25.7617,
        "nws_lon":      -80.1918,
        "iem_station":  "MIA",
        "iem_network":  "FL_ASOS",
    },
    {
        "key":          "atlanta",
        "display":      "Atlanta",
        "slug_city":    "atlanta",
        "nws_lat":      33.6407,
        "nws_lon":      -84.4277,
        "iem_station":  "ATL",
        "iem_network":  "GA_ASOS",
    },
    {
        "key":          "chicago",
        "display":      "Chicago",
        "slug_city":    "chicago",
        "nws_lat":      41.9742,
        "nws_lon":      -87.9073,
        "iem_station":  "ORD",
        "iem_network":  "IL_ASOS",
    },
    {
        "key":          "seattle",
        "display":      "Seattle",
        "slug_city":    "seattle",
        "nws_lat":      47.4502,
        "nws_lon":      -122.3088,
        "iem_station":  "SEA",
        "iem_network":  "WA_ASOS",
    },
    {
        "key":          "dallas",
        "display":      "Dallas",
        "slug_city":    "dallas",
        "nws_lat":      32.8998,
        "nws_lon":      -97.0403,
        "iem_station":  "DFW",
        "iem_network":  "TX_ASOS",
    },
]

# NWS grid point cache — keyed by city key
_GRID_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

def _market_slug(target_date: date, city: dict) -> str:
    """
    e.g. date(2026,2,28), "nyc" → "highest-temperature-in-nyc-on-february-28-2026"
    Leading zeroes are omitted from the day (Polymarket convention).
    """
    month = target_date.strftime("%B").lower()
    day   = str(target_date.day)
    year  = target_date.strftime("%Y")
    return f"highest-temperature-in-{city['slug_city']}-on-{month}-{day}-{year}"


# ---------------------------------------------------------------------------
# Bracket parsing and implied temperature
# ---------------------------------------------------------------------------

def _extract_bracket(question: str) -> str | None:
    m = re.search(r'(\d+)°F or (below|higher)', question)
    if m:
        return f"{m.group(1)}°F or {m.group(2)}"
    m = re.search(r'between (\d+-\d+)°F', question)
    if m:
        return f"{m.group(1)}°F"
    return None


def _bracket_midpoint(bracket: str) -> float | None:
    b = bracket.strip()
    if "or below" in b:
        try:
            return float(b.split("°")[0]) - 1.0
        except ValueError:
            return None
    if "or higher" in b:
        try:
            return float(b.split("°")[0]) + 1.0
        except ValueError:
            return None
    if "-" in b and "°F" in b:
        try:
            parts = b.replace("°F", "").split("-")
            return (float(parts[0]) + float(parts[1])) / 2.0
        except (ValueError, IndexError):
            return None
    return None


def _market_implied_temp(brackets: list[str], prices: list[float]) -> float | None:
    total_prob = 0.0
    weighted   = 0.0
    for bracket, p in zip(brackets, prices):
        mid = _bracket_midpoint(bracket)
        if mid is not None:
            weighted   += mid * p
            total_prob += p
    return (weighted / total_prob) if total_prob > 0 else None


# ---------------------------------------------------------------------------
# Polymarket Gamma API
# ---------------------------------------------------------------------------

def fetch_city_market(target_date: date, city: dict) -> dict | None:
    """
    Fetch Polymarket event data for a city's daily high temp on *target_date*.
    Returns None if no event exists yet.
    """
    slug = _market_slug(target_date, city)
    url  = f"{GAMMA_API}/events?slug={slug}"

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        events = resp.json()
    except Exception as exc:
        logger.warning("Polymarket fetch failed %s / %s: %s", city["key"], slug, exc)
        return None

    if not events:
        return None

    event   = events[0]
    markets = event.get("markets", [])
    if not markets:
        return None

    brackets = []
    prices   = []
    resolved = False
    winning_bracket = None
    total_volume = 0.0

    for m in markets:
        bracket = _extract_bracket(m.get("question", ""))
        if bracket is None:
            continue

        raw = m.get("outcomePrices", "[]")
        if isinstance(raw, str):
            try:
                p_list = [float(p) for p in json.loads(raw)]
            except (json.JSONDecodeError, ValueError):
                p_list = []
        else:
            p_list = [float(p) for p in raw]

        p_yes = p_list[0] if p_list else 0.0
        brackets.append(bracket)
        prices.append(p_yes)

        if m.get("resolved", False) or p_yes >= 0.99:
            resolved = True
            if p_yes >= 0.99:
                winning_bracket = bracket

        try:
            total_volume += float(m.get("volume", 0) or 0)
        except (TypeError, ValueError):
            pass

    if not brackets:
        return None

    return {
        "slug":            slug,
        "question":        event.get("title", ""),
        "brackets":        brackets,
        "prices":          prices,
        "resolved":        resolved,
        "winning_bracket": winning_bracket,
        "volume":          total_volume,
    }


# ---------------------------------------------------------------------------
# NWS hourly forecast — per-city daily high
# ---------------------------------------------------------------------------

def _get_nws_grid(city: dict) -> tuple:
    key = city["key"]
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]

    url  = f"{NWS_API}/points/{city['nws_lat']},{city['nws_lon']}"
    resp = requests.get(url, headers=NWS_HEADERS, timeout=15)
    resp.raise_for_status()
    props = resp.json()["properties"]
    result = (props["gridId"], props["gridX"], props["gridY"], props["forecastHourly"])
    _GRID_CACHE[key] = result
    return result


def get_nws_daily_high(target_date: date, city: dict) -> float | None:
    """NWS hourly forecast max temperature for *target_date* at *city* in °F."""
    try:
        _, _, _, hourly_url = _get_nws_grid(city)
        resp = requests.get(hourly_url, headers=NWS_HEADERS, timeout=15)
        resp.raise_for_status()
        periods = resp.json()["properties"]["periods"]
    except Exception as exc:
        logger.warning("NWS hourly fetch failed for %s: %s", city["key"], exc)
        return None

    daily_temps = []
    for p in periods:
        try:
            dt = datetime.fromisoformat(p["startTime"])
            if dt.date() == target_date:
                daily_temps.append(p["temperature"])
        except (ValueError, KeyError):
            continue

    return float(max(daily_temps)) if daily_temps else None


# ---------------------------------------------------------------------------
# IEM ASOS — actual observed daily high (for past dates)
# ---------------------------------------------------------------------------

def fetch_observed_high(target_date: date, city: dict) -> float | None:
    """
    Fetch the actual observed daily max temperature (°F) from IEM ASOS for *city*.

    IEM's daily.json requires a network parameter and always returns the full
    month regardless of the day parameter, so we filter by date in the response.
    """
    station  = city["iem_station"]
    network  = city["iem_network"]
    date_str = target_date.isoformat()
    url = (
        "https://mesonet.agron.iastate.edu/api/1/daily.json"
        f"?station={station}&network={network}"
        f"&year={target_date.year}&month={target_date.month}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        records = resp.json().get("data", [])
        for rec in records:
            if rec.get("date") == date_str:
                val = rec.get("max_tmpf")
                if val is not None:
                    return float(val)
    except Exception as exc:
        logger.warning("IEM daily fetch failed for %s/%s %s: %s", station, network, target_date, exc)
    return None


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def load_polymarket_cache() -> dict:
    if POLYMARKET_CACHE_PATH.exists():
        try:
            with open(POLYMARKET_CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to load polymarket cache: %s", exc)
    return {}


def save_polymarket_cache(data: dict) -> None:
    try:
        POLYMARKET_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(POLYMARKET_CACHE_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to save polymarket cache: %s", exc)


# ---------------------------------------------------------------------------
# Main update function
# ---------------------------------------------------------------------------

def update_polymarket_cache() -> dict:
    """
    For each city, fetch up to 6 days of Polymarket markets and NWS forecasts.
    Back-fills observed highs from IEM ASOS for past dates.
    Returns summary dict.
    """
    cache = load_polymarket_cache()
    today = date.today()
    summary = {}

    for city in CITIES:
        city_key = city["key"]
        city_cache = cache.setdefault(city_key, {"markets": {}})
        markets = city_cache.setdefault("markets", {})
        found = updated = 0

        # ---- Snapshot at sync time for upcoming markets ----
        now_utc  = datetime.now(timezone.utc)
        now_str  = now_utc.isoformat()

        for days_ahead in range(6):
            target   = today + timedelta(days=days_ahead)
            date_str = target.isoformat()

            market_data = fetch_city_market(target, city)
            if market_data is None:
                continue

            found   += 1
            nws_high = get_nws_daily_high(target, city)
            implied  = _market_implied_temp(market_data["brackets"], market_data["prices"])

            # Lead time in hours (float) so 6-hourly syncs produce distinct points.
            # Target "event time" is midnight UTC of the following day (i.e. end of
            # the target calendar day in Eastern time ≈ 05:00 UTC next day, but
            # midnight UTC is a consistent, simple reference).
            target_midnight_utc = datetime(target.year, target.month, target.day,
                                           tzinfo=timezone.utc) + timedelta(days=1)
            lead_hours = (target_midnight_utc - now_utc).total_seconds() / 3600

            snapshot = {
                "fetched_at":          now_str,
                "lead_hours":          round(lead_hours, 2),
                "market_implied_temp": implied,
                "nws_high":            nws_high,
                "volume":              market_data["volume"],
            }

            if date_str not in markets:
                markets[date_str] = {
                    "slug":            market_data["slug"],
                    "question":        market_data["question"],
                    "brackets":        market_data["brackets"],
                    "resolved":        market_data["resolved"],
                    "winning_bracket": market_data["winning_bracket"],
                    "observed_high":   None,
                    "snapshots":       [],
                }

            entry = markets[date_str]
            entry["question"]        = market_data["question"]
            entry["resolved"]        = market_data["resolved"]
            entry["winning_bracket"] = market_data["winning_bracket"]
            entry["brackets"]        = market_data["brackets"]
            entry["snapshots"].append(snapshot)
            updated += 1

            logger.info(
                "Polymarket %s %s: implied=%.1f°F  nws=%s°F  lead=%.1fh",
                city_key, date_str,
                implied if implied is not None else float("nan"),
                f"{nws_high:.1f}" if nws_high is not None else "—",
                lead_hours,
            )

        # ---- Re-check resolution for recent past markets ----
        # The live-market loop only covers today + 5 days forward, so markets
        # from yesterday or earlier never get their resolved flag updated.
        # Check the last 7 days for any that resolved since we last looked.
        for date_str, entry in markets.items():
            target = date.fromisoformat(date_str)
            if target >= today:
                continue   # still active, handled above
            if entry.get("resolved"):
                continue   # already know it resolved
            age_days = (today - target).days
            if age_days > 7:
                continue   # too old, assume we'd have caught it by now
            market_data = fetch_city_market(target, city)
            if market_data and market_data["resolved"]:
                entry["resolved"]        = True
                entry["winning_bracket"] = market_data["winning_bracket"]
                logger.info("Resolved %s %s → %s", city_key, date_str, market_data["winning_bracket"])

        # ---- Back-fill observed highs for past markets ----
        for date_str, entry in markets.items():
            target = date.fromisoformat(date_str)
            if target >= today or entry.get("observed_high") is not None:
                continue
            obs = fetch_observed_high(target, city)
            if obs is not None:
                entry["observed_high"] = obs
                logger.info("Filled observed high %s %s: %.1f°F", city_key, date_str, obs)

        summary[city_key] = {"found": found, "updated": updated}

    # Also expose city metadata in cache for the frontend
    cache["_cities"] = [{"key": c["key"], "display": c["display"]} for c in CITIES]

    save_polymarket_cache(cache)
    return summary


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("Updating Polymarket cache (all cities)...")
    result = update_polymarket_cache()
    print(f"\nSummary: {result}")

    cache = load_polymarket_cache()
    for city in CITIES:
        city_key = city["key"]
        city_data = cache.get(city_key, {})
        markets = city_data.get("markets", {})
        print(f"\n{'='*50}")
        print(f"{city['display']} ({city_key}): {len(markets)} dates")
        for date_str in sorted(markets):
            entry = markets[date_str]
            snap  = entry["snapshots"][-1] if entry["snapshots"] else {}
            imp   = snap.get("market_implied_temp")
            nws   = snap.get("nws_high")
            print(f"  {date_str}:  implied={f'{imp:.1f}°F' if imp else '—':>8}  "
                  f"nws={f'{nws:.1f}°F' if nws else '—':>8}  "
                  f"lead={snap.get('lead_hours', '?'):.1f}h" if snap.get('lead_hours') is not None else "lead=—")
