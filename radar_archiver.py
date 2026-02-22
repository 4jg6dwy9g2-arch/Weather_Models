#!/usr/bin/env python3
"""
Radar frame archiver — run every 5 minutes via launchd.

Archives two types of radar products:
  1. MRMS PrecipRate_00.00 — downloaded as GRIB2 from NOAA AWS S3, rendered
     to PNG with NWS precipitation rate colormap.
  2. NWS GeoServer WMS products — single-site KLWX/TIAD layers fetched as
     pre-rendered PNG images directly from opengeo.ncep.noaa.gov.

All frames stored under DATA_DIR/radar_frames/<product_key>/ with an index.json.
Retention: 24 hours (~720 frames per product).

Install:
    cp com.weathermodels.radar-archiver.plist ~/Library/LaunchAgents/
    launchctl load ~/Library/LaunchAgents/com.weathermodels.radar-archiver.plist
    launchctl list | grep radar-archiver
"""

import fcntl
import gzip
import json
import logging
import math
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR      = Path("/Volumes/T7/Weather_Models/data")
RADAR_DIR     = DATA_DIR / "radar_frames"
LIGHTNING_DIR = DATA_DIR / "lightning"
LOG_DIR       = DATA_DIR / "logs"
LOCK_FILE     = "/tmp/radar_archiver.lock"

# Convenience alias for the PrecipRate subdirectory
PRECIP_DIR = RADAR_DIR / "precip"

# ── Constants ─────────────────────────────────────────────────────────────────

MRMS_S3_BASE  = "https://noaa-mrms-pds.s3.amazonaws.com"
MRMS_PRODUCT  = "PrecipRate_00.00"
RETENTION_HRS = 24  # ~288 frames per product at 5-min intervals
WMS_BASE      = "https://opengeo.ncep.noaa.gov/geoserver"
LOOP_RENDER_STEPS = (4, 2, 1)
DEFAULT_LOOP_STEP = 4
WMS_STEP_SCALE = {
    4: 1.0,
    2: 2.0,
    1: 3.0,
}

# GOES-R GLM lightning
GLM_BUCKETS = {
    "goes19": "https://noaa-goes19.s3.amazonaws.com",  # GOES-East
    "goes18": "https://noaa-goes18.s3.amazonaws.com",  # GOES-West
}
GLM_STATE_FILE   = LIGHTNING_DIR / "archiver_state.json"
GLM_WINDOW_MIN   = 4   # look back 4 min (archiver runs every 2 min; small buffer)

# Fairfax, VA — home location for proximity tracking
FAIRFAX_LAT         = 38.846
FAIRFAX_LON         = -77.307
FAIRFAX_RADIUS_MI   = 50    # miles — alert threshold
ALERT_COOLDOWN_MIN  = 10    # minimum minutes between alerts
ALERT_RECIPIENT     = os.environ.get("IMESSAGE_RECIPIENT", "")
ALERT_LAST_FILE     = LIGHTNING_DIR / "last_alert.json"

# WMS products: each entry defines the native geographic bounds.
# bbox format: "minLon,minLat,maxLon,maxLat" (degrees).
WMS_PRODUCTS = {
    "klwx_sr_bref": {
        "url":    f"{WMS_BASE}/klwx/ows",
        "layer":  "klwx_sr_bref",
        "bbox":   "-82.487,33.977,-72.488,43.976",
        "width":  1200,
        "height": 1200,
    },
    "klwx_sr_bvel": {
        "url":    f"{WMS_BASE}/klwx/ows",
        "layer":  "klwx_sr_bvel",
        "bbox":   "-82.487,33.977,-72.488,43.976",
        "width":  1200,
        "height": 1200,
    },
    "klwx_bdhc": {
        "url":    f"{WMS_BASE}/klwx/ows",
        "layer":  "klwx_bdhc",
        "bbox":   "-82.487,33.977,-72.488,43.976",
        "width":  1200,
        "height": 1200,
    },
    "tiad_bref1": {
        "url":    f"{WMS_BASE}/tiad/ows",
        "layer":  "tiad_bref1",
        "bbox":   "-78.529,38.084,-76.529,40.084",
        "width":  800,
        "height": 800,
    },
    "tiad_brefl": {
        "url":    f"{WMS_BASE}/tiad/ows",
        "layer":  "tiad_brefl",
        "bbox":   "-81.529,35.084,-73.529,43.084",
        "width":  1000,
        "height": 1000,
    },
    "tiad_bvel": {
        "url":    f"{WMS_BASE}/tiad/ows",
        "layer":  "tiad_bvel",
        "bbox":   "-78.529,38.084,-76.529,40.084",
        "width":  800,
        "height": 800,
    },
}


def _lonlat_to_web_mercator(lon: float, lat: float) -> tuple[float, float]:
    """Convert lon/lat degrees to EPSG:3857 meters."""
    # Clamp latitude to Mercator's practical limits
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) * 20037508.34 / math.pi
    return x, y


def _bbox_lonlat_to_web_mercator(bbox_str: str) -> str:
    """Convert 'minLon,minLat,maxLon,maxLat' to EPSG:3857 BBOX string."""
    min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox_str.split(",")]
    min_x, min_y = _lonlat_to_web_mercator(min_lon, min_lat)
    max_x, max_y = _lonlat_to_web_mercator(max_lon, max_lat)
    return f"{min_x:.2f},{min_y:.2f},{max_x:.2f},{max_y:.2f}"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)


# ── iMessage alerts ───────────────────────────────────────────────────────────

def _send_imessage(recipient: str, message: str) -> None:
    """Send an iMessage via Messages.app using osascript."""
    if not recipient:
        return
    script = f'''tell application "Messages"
        set targetService to first service whose service type is iMessage
        set targetBuddy to buddy "{recipient}" of targetService
        send "{message}" to targetBuddy
    end tell'''
    try:
        import subprocess
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True, text=True)
        logger.info("iMessage sent to %s", recipient)
    except Exception as e:
        logger.warning("iMessage failed: %s", e)


def _maybe_send_lightning_alert(nearby: list) -> None:
    """Send an iMessage if nearby lightning was detected and cooldown has elapsed."""
    if not ALERT_RECIPIENT or not nearby:
        return

    now = datetime.now(timezone.utc)

    # Check cooldown
    if ALERT_LAST_FILE.exists():
        try:
            last = datetime.fromisoformat(json.loads(ALERT_LAST_FILE.read_text())["t"])
            if (now - last).total_seconds() < ALERT_COOLDOWN_MIN * 60:
                logger.info("Lightning alert suppressed (cooldown active, last: %s)", last.isoformat())
                return
        except Exception:
            pass

    # Build message
    distances = sorted(
        _haversine_miles(FAIRFAX_LAT, FAIRFAX_LON, f["lat"], f["lon"])
        for f in nearby
    )
    closest = distances[0]
    count   = len(nearby)
    t_local = now.astimezone()  # local time for display
    time_str = t_local.strftime("%-I:%M %p")

    msg = (
        f"⚡ Lightning Alert ({time_str}): "
        f"{count} flash{'es' if count != 1 else ''} within {FAIRFAX_RADIUS_MI} mi of Fairfax. "
        f"Closest: {closest:.0f} mi."
    )
    _send_imessage(ALERT_RECIPIENT, msg)

    # Save last-alert timestamp
    LIGHTNING_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write(ALERT_LAST_FILE, json.dumps({"t": now.isoformat()}).encode())


# ── Lock ──────────────────────────────────────────────────────────────────────

def _acquire_lock():
    """Acquire exclusive flock; exit immediately if another instance holds it."""
    f = open(LOCK_FILE, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        logger.info("Another instance is already running — exiting.")
        sys.exit(0)
    return f


# ── Atomic write ──────────────────────────────────────────────────────────────

def _atomic_write(dest: Path, data: bytes) -> None:
    """Write data to a .tmp file then rename — safe against mid-write kills."""
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.rename(dest)


# ── Generic index helpers (work for any product directory) ────────────────────

def _read_index(frames_dir: Path) -> dict:
    index_file = frames_dir / "index.json"
    if index_file.exists():
        try:
            return json.loads(index_file.read_text())
        except Exception:
            pass
    return {"product": frames_dir.name, "updated": None, "frames": []}


def _step_filename(base_filename: str, step: int) -> str:
    """Map a base frame filename to a step-specific variant filename."""
    if int(step) == DEFAULT_LOOP_STEP:
        return base_filename
    stem, ext = os.path.splitext(base_filename)
    return f"{stem}_s{int(step)}{ext}"


def _last_archived_time(frames_dir: Path):
    """Return the most recently archived frame's valid_time, or None."""
    frames = _read_index(frames_dir).get("frames", [])
    if not frames:
        return None
    try:
        return datetime.fromisoformat(frames[-1]["t"])
    except Exception:
        return None


def _update_index(frames_dir: Path, valid_time: datetime, filename: str, variants: dict | None = None) -> None:
    idx    = _read_index(frames_dir)
    frames = idx.get("frames", [])
    t_iso  = valid_time.isoformat()
    variants = variants or {}

    for frame in frames:
        if frame.get("t") != t_iso:
            continue
        frame["f"] = frame.get("f") or filename
        if variants:
            s_map = frame.get("s", {})
            for k, v in variants.items():
                s_map[str(k)] = v
            frame["s"] = s_map
        idx["updated"] = datetime.now(timezone.utc).isoformat()
        _atomic_write(frames_dir / "index.json", json.dumps(idx, indent=2).encode())
        return

    entry = {"t": t_iso, "f": filename}
    if variants:
        entry["s"] = {str(k): v for k, v in variants.items()}
    frames.append(entry)
    frames.sort(key=lambda x: x["t"])
    idx["frames"]  = frames
    idx["updated"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(frames_dir / "index.json", json.dumps(idx, indent=2).encode())


def _prune_old_frames(frames_dir: Path) -> None:
    idx    = _read_index(frames_dir)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=RETENTION_HRS)
    kept, removed = [], []

    for f in idx.get("frames", []):
        try:
            t = datetime.fromisoformat(f["t"])
        except Exception:
            kept.append(f)
            continue
        (kept if t >= cutoff else removed).append(f)

    for f in removed:
        to_delete = []
        base_name = f.get("f")
        if base_name:
            to_delete.append(base_name)
        for name in (f.get("s") or {}).values():
            if name:
                to_delete.append(name)
        for name in sorted(set(to_delete)):
            try:
                (frames_dir / name).unlink(missing_ok=True)
                logger.info("Pruned %s/%s", frames_dir.name, name)
            except Exception as e:
                logger.warning("Could not delete %s: %s", name, e)

    if removed:
        idx["frames"]  = kept
        idx["updated"] = datetime.now(timezone.utc).isoformat()
        _atomic_write(frames_dir / "index.json", json.dumps(idx, indent=2).encode())
        logger.info("%s: pruned %d old frames, %d remain", frames_dir.name, len(removed), len(kept))


# ── MRMS S3 helpers ───────────────────────────────────────────────────────────

def _list_s3_keys(prefix: str) -> list:
    url = f"{MRMS_S3_BASE}?prefix={prefix}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(r.text)
    return [el.text for el in root.findall(".//s3:Key", ns) if el.text]


def _find_latest_precip_key() -> tuple:
    """Return (s3_key, valid_time) for the most recent PrecipRate GRIB2.GZ."""
    now = datetime.now(timezone.utc)
    for day_delta in range(2):
        day    = now - timedelta(days=day_delta)
        prefix = f"CONUS/{MRMS_PRODUCT}/{day.strftime('%Y%m%d')}/"
        try:
            keys = [k for k in _list_s3_keys(prefix) if k.endswith(".grib2.gz")]
        except Exception as e:
            logger.warning("S3 listing failed for %s: %s", prefix, e)
            continue
        if not keys:
            continue

        key   = sorted(keys)[-1]
        fname = key.rsplit("/", 1)[-1]
        try:
            ts_raw     = fname.split("_")[-1]
            ts_raw     = ts_raw.split(".grib2")[0]
            ts_raw     = ts_raw.rsplit(".", 1)[0]
            valid_time = datetime.strptime(ts_raw, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
        except Exception:
            valid_time = None

        return key, valid_time

    return None, None


# ── MRMS PrecipRate archiving ─────────────────────────────────────────────────

def _archive_mrms_precip() -> None:
    """Download the latest MRMS PrecipRate frame if not already archived."""
    PRECIP_DIR.mkdir(parents=True, exist_ok=True)

    key, valid_time = _find_latest_precip_key()
    if key is None:
        logger.error("No PrecipRate files found on S3")
        return

    logger.info("PrecipRate latest S3 key: %s  valid: %s", key, valid_time)

    last = _last_archived_time(PRECIP_DIR)
    if last is not None and valid_time is not None and last >= valid_time:
        logger.info("PrecipRate already current (%s) — skipping", last.isoformat())
        return

    # Import radar here (after lock acquired) to avoid startup overhead on early exits
    import radar

    url = f"{MRMS_S3_BASE}/{key}"
    logger.info("Fetching %s", url)
    t0 = time.time()

    r = requests.get(url, timeout=90)
    r.raise_for_status()

    grib_data = gzip.decompress(r.content)
    logger.info("Decompressed %.1f MB in %.1fs", len(grib_data) / 1e6, time.time() - t0)

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(grib_data)
        tmp_path = tmp.name

    try:
        data = radar.read_grib2(tmp_path)
        png_by_step = {
            step: radar.render_precip_png(data, step=step)
            for step in LOOP_RENDER_STEPS
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    filename = valid_time.strftime("%Y%m%d_%H%M%S") + ".png"
    variants = {}
    for step, png_bytes in png_by_step.items():
        step_name = _step_filename(filename, step)
        _atomic_write(PRECIP_DIR / step_name, png_bytes)
        variants[str(step)] = step_name
    logger.info("PrecipRate saved %s (%s tiers) in %.1fs total",
                filename, ",".join(str(s) for s in LOOP_RENDER_STEPS), time.time() - t0)

    _update_index(PRECIP_DIR, valid_time, filename, variants=variants)
    _prune_old_frames(PRECIP_DIR)


# ── WMS archiving ─────────────────────────────────────────────────────────────

def _archive_wms_product(product_key: str, cfg: dict, snap_time: datetime) -> None:
    """Download one WMS GetMap frame and add it to the product's archive."""
    frames_dir = RADAR_DIR / product_key
    frames_dir.mkdir(parents=True, exist_ok=True)

    last = _last_archived_time(frames_dir)
    if last is not None and last >= snap_time:
        logger.info("WMS %s already current (%s) — skipping", product_key, last.isoformat())
        return

    t0 = time.time()
    filename = snap_time.strftime("%Y%m%d_%H%M%S") + ".png"
    variants = {}
    mercator_bbox = _bbox_lonlat_to_web_mercator(cfg["bbox"])
    for step in LOOP_RENDER_STEPS:
        scale = WMS_STEP_SCALE.get(step, 1.0)
        width = max(256, int(round(cfg["width"] * scale)))
        height = max(256, int(round(cfg["height"] * scale)))
        params = {
            "SERVICE":     "WMS",
            "VERSION":     "1.1.1",
            "REQUEST":     "GetMap",
            "LAYERS":      cfg["layer"],
            "BBOX":        mercator_bbox,
            "SRS":         "EPSG:3857",
            "WIDTH":       width,
            "HEIGHT":      height,
            "FORMAT":      "image/png",
            "TRANSPARENT": "true",
        }
        r = requests.get(cfg["url"], params=params, timeout=45,
                         headers={"User-Agent": "WeatherModels/1.0"})
        r.raise_for_status()

        ct = r.headers.get("Content-Type", "")
        if not ct.startswith("image/"):
            raise ValueError(f"WMS {product_key}: expected image, got {ct!r}: {r.text[:200]}")

        step_name = _step_filename(filename, step)
        _atomic_write(frames_dir / step_name, r.content)
        variants[str(step)] = step_name
    logger.info("WMS %s saved %s (%s tiers) in %.1fs",
                product_key, filename, ",".join(str(s) for s in LOOP_RENDER_STEPS), time.time() - t0)

    _update_index(frames_dir, snap_time, filename, variants=variants)
    _prune_old_frames(frames_dir)


# ── GLM lightning helpers ─────────────────────────────────────────────────────

def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _glm_key_time(key: str) -> datetime:
    """Parse valid_time from GLM S3 key (OR_GLM-L2-LCFA_Gxx_sYYYYDDDHHMMSSf_…)."""
    fname = key.rsplit("/", 1)[-1]
    start = fname.split("_s")[1][:13]
    return datetime.strptime(start, "%Y%j%H%M%S").replace(tzinfo=timezone.utc)


def _list_glm_keys(satellite: str, cutoff: datetime) -> list:
    """Return GLM S3 keys >= cutoff for the given satellite, sorted ascending."""
    bucket = GLM_BUCKETS[satellite]
    ns     = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    now    = datetime.now(timezone.utc)

    prefixes, t = set(), cutoff
    while t <= now + timedelta(minutes=1):
        doy = t.timetuple().tm_yday
        prefixes.add(f"GLM-L2-LCFA/{t.year}/{doy:03d}/{t.hour:02d}/")
        t += timedelta(hours=1)

    all_keys = []
    for prefix in sorted(prefixes):
        try:
            r = requests.get(f"{bucket}?prefix={prefix}", timeout=10)
            root = ET.fromstring(r.text)
            all_keys.extend(
                el.text for el in root.findall(".//s3:Key", ns)
                if el.text and el.text.endswith(".nc")
            )
        except Exception as e:
            logger.warning("GLM key listing failed for %s %s: %s", satellite, prefix, e)

    return sorted(k for k in all_keys if _glm_key_time(k) >= cutoff)


def _parse_glm_flashes(url: str, file_start: datetime) -> list:
    """Download one GLM NetCDF and return CONUS flash dicts (lat, lon, t)."""
    import netCDF4
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as fh:
        fh.write(r.content)
        tmp = fh.name
    try:
        ds      = netCDF4.Dataset(tmp)
        lats    = ds.variables["flash_lat"][:].tolist()
        lons    = ds.variables["flash_lon"][:].tolist()
        offsets = ds.variables["flash_time_offset_of_first_event"][:].tolist()
        ds.close()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    flashes = []
    for lat, lon, off in zip(lats, lons, offsets):
        # Filter to broad CONUS + adjacent ocean/Canada region
        if 18.0 <= lat <= 58.0 and -135.0 <= lon <= -55.0:
            t = file_start + timedelta(seconds=float(off))
            flashes.append({
                "lat": round(float(lat), 4),
                "lon": round(float(lon), 4),
                "t":   t.isoformat(),
            })
    return flashes


def _archive_lightning() -> None:
    """Fetch new GLM files, append CONUS flashes to daily JSONL, track Fairfax proximity."""
    LIGHTNING_DIR.mkdir(parents=True, exist_ok=True)

    # Load per-satellite state (tracks last processed S3 key)
    state: dict = {"goes19": None, "goes18": None}
    if GLM_STATE_FILE.exists():
        try:
            state.update(json.loads(GLM_STATE_FILE.read_text()))
        except Exception:
            pass

    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=GLM_WINDOW_MIN)
    today  = now.strftime("%Y%m%d")

    all_new: list  = []
    nearby:  list  = []

    for satellite in ("goes19", "goes18"):
        bucket   = GLM_BUCKETS[satellite]
        last_key = state.get(satellite)

        try:
            keys = _list_glm_keys(satellite, cutoff)
        except Exception as e:
            logger.error("GLM key listing failed for %s: %s", satellite, e)
            continue

        new_keys = [k for k in keys if last_key is None or k > last_key]
        for key in new_keys:
            try:
                file_start = _glm_key_time(key)
                flashes    = _parse_glm_flashes(f"{bucket}/{key}", file_start)
                all_new.extend(flashes)
                logger.info("GLM %s: %d CONUS flashes from %s",
                            satellite, len(flashes), key.rsplit("/", 1)[-1])
            except Exception as e:
                logger.warning("GLM parse failed for %s %s: %s", satellite, key, e)

        if new_keys:
            state[satellite] = new_keys[-1]

    if all_new:
        # Append to today's JSONL archive
        jsonl = LIGHTNING_DIR / f"{today}.jsonl"
        with open(jsonl, "a") as fh:
            for f in all_new:
                fh.write(json.dumps(f) + "\n")
        logger.info("Lightning: appended %d CONUS flashes to %s", len(all_new), jsonl.name)

        # Fairfax proximity check
        nearby = [
            f for f in all_new
            if _haversine_miles(FAIRFAX_LAT, FAIRFAX_LON, f["lat"], f["lon"]) <= FAIRFAX_RADIUS_MI
        ]
        if nearby:
            logger.info("⚡ %d flash(es) within %d mi of Fairfax!", len(nearby), FAIRFAX_RADIUS_MI)
            # Update rolling 60-minute Fairfax proximity file
            prox_file = LIGHTNING_DIR / "fairfax_proximity.json"
            existing: list = []
            if prox_file.exists():
                try:
                    existing = json.loads(prox_file.read_text())
                except Exception:
                    pass
            cutoff_60 = (now - timedelta(minutes=60)).isoformat()
            existing  = [f for f in existing if f["t"] >= cutoff_60]
            existing.extend(nearby)
            _atomic_write(prox_file, json.dumps(existing).encode())
            _maybe_send_lightning_alert(nearby)

    # Save state and prune old JSONL files (keep last 2 days)
    _atomic_write(GLM_STATE_FILE, json.dumps(state).encode())
    for old in sorted(LIGHTNING_DIR.glob("*.jsonl"))[:-2]:
        try:
            old.unlink()
            logger.info("Pruned old lightning file: %s", old.name)
        except OSError:
            pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0      = time.time()
    lock_fh = _acquire_lock()

    RADAR_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 1. MRMS PrecipRate
        try:
            _archive_mrms_precip()
        except Exception as e:
            logger.error("PrecipRate archiver failed: %s", e, exc_info=True)

        # 2. WMS products — all share the same rounded snapshot timestamp so
        #    frames across products are aligned to the same 2-minute boundary.
        now      = datetime.now(timezone.utc)
        snap     = now.replace(second=0, microsecond=0)
        snap     = snap - timedelta(minutes=snap.minute % 2)

        for key, cfg in WMS_PRODUCTS.items():
            try:
                _archive_wms_product(key, cfg, snap)
            except Exception as e:
                logger.error("WMS %s failed: %s", key, e, exc_info=True)

        # 3. GLM lightning
        try:
            _archive_lightning()
        except Exception as e:
            logger.error("Lightning archiver failed: %s", e, exc_info=True)

        logger.info("All products done in %.1fs", time.time() - t0)

    finally:
        lock_fh.close()


if __name__ == "__main__":
    main()
