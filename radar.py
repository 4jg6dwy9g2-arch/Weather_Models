"""
NEXRAD MRMS national composite radar from NOAA AWS Open Data.

Downloads the latest MergedReflectivityQCComposite_00.50 GRIB2 file from the
noaa-mrms-pds S3 bucket (public, no credentials needed), renders it to a PNG
with the NWS standard reflectivity colormap, and caches the result in memory.
A daemon thread refreshes the cache every CACHE_TTL seconds.

MRMS CONUS domain (fixed grid):
  3500 rows × 7000 cols, 0.01° resolution
  Lat: 20.005°N – 54.995°N (row 0 = northernmost, matching image convention)
  Lon: 130.005°W – 60.005°W
  Updated every 2 minutes on AWS.
"""

import gzip
import io
import logging
import os
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MRMS_S3_BASE  = "https://noaa-mrms-pds.s3.amazonaws.com"
MRMS_PRODUCT  = "MergedReflectivityQCComposite_00.50"
CACHE_TTL     = 120   # seconds; MRMS updates every ~2 min
RENDER_STEP   = 4     # downsample factor: 3500→875 rows, 7000→1750 cols (~4 km)
RENDER_STEPS  = (4, 2, 1)  # Coarse→fine; step 1 preserves native 0.01° grid detail

# Fixed MRMS CONUS bounds — used to set the Leaflet imageOverlay extent.
# La1 / La2 / Lo1 / Lo2 as defined in the MRMS GRIB2 product specification.
MRMS_BOUNDS = {
    "south": 20.005,
    "north": 54.995,
    "west":  -130.005,
    "east":  -60.005,
}

# NWS standard WSR-88D reflectivity color scale.
# Each entry: (lower_dbz_threshold, R, G, B)
# Pixels below 5 dBZ (no echo) are fully transparent.
_NWS_CMAP = [
    ( 5, 0x04, 0xe9, 0xe7),   # light cyan
    (10, 0x01, 0x9f, 0xf4),   # sky blue
    (15, 0x02, 0x11, 0xd4),   # deep blue
    (20, 0x02, 0xfd, 0x02),   # bright green
    (25, 0x01, 0xc5, 0x01),   # medium green
    (30, 0x00, 0x83, 0x01),   # dark green
    (35, 0xff, 0xff, 0x00),   # yellow
    (40, 0xe7, 0xc0, 0x00),   # amber
    (45, 0xff, 0x90, 0x00),   # orange
    (50, 0xff, 0x00, 0x00),   # red
    (55, 0xd4, 0x00, 0x00),   # dark red
    (60, 0xc0, 0x00, 0x00),   # very dark red
    (65, 0xff, 0x00, 0xff),   # magenta
    (70, 0x99, 0x55, 0xc9),   # purple
    (75, 0xff, 0xff, 0xff),   # white (extreme)
]

# ── In-memory cache ───────────────────────────────────────────────────────────

_cache_lock = threading.Lock()
_cache: dict = {
    "png_bytes":  None,   # bytes | None
    "png_by_step": {},    # dict[int, bytes]
    "valid_time": None,   # datetime | None
    "fetched_at": 0.0,    # unix timestamp of last successful fetch
    "error":      None,   # str | None  (last error message)
}

# ── S3 helpers ────────────────────────────────────────────────────────────────

def _list_s3_keys(prefix: str) -> list[str]:
    """Return all object keys in noaa-mrms-pds matching the given prefix."""
    url = f"{MRMS_S3_BASE}?prefix={prefix}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(r.text)
    return [el.text for el in root.findall(".//s3:Key", ns) if el.text]


def _find_latest_key() -> tuple:
    """
    Return (s3_key, valid_time) for the most recent MRMS GRIB2.GZ file.
    Searches today and yesterday in case the day just rolled over.
    """
    now = datetime.now(timezone.utc)
    for day_delta in range(2):
        day = now - timedelta(days=day_delta)
        prefix = f"CONUS/{MRMS_PRODUCT}/{day.strftime('%Y%m%d')}/"
        try:
            keys = [k for k in _list_s3_keys(prefix) if k.endswith(".grib2.gz")]
        except Exception as e:
            logger.warning("S3 listing failed for %s: %s", prefix, e)
            continue
        if not keys:
            continue

        key = sorted(keys)[-1]

        # Filename: MRMS_MergedReflectivityQCComposite_00.50_YYYYMMDD-HHMMSS.00.grib2.gz
        fname = key.rsplit("/", 1)[-1]
        try:
            ts_raw = fname.split("_")[-1]          # "20260216-183000.00.grib2.gz"
            ts_raw = ts_raw.split(".grib2")[0]      # "20260216-183000.00"
            ts_raw = ts_raw.rsplit(".", 1)[0]       # "20260216-183000"
            valid_time = datetime.strptime(ts_raw, "%Y%m%d-%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except Exception:
            valid_time = None

        return key, valid_time

    return None, None


# ── GRIB2 reading ─────────────────────────────────────────────────────────────

def _read_with_eccodes(path: str) -> np.ndarray:
    """
    Read reflectivity values from a MRMS GRIB2 file using eccodes.
    Returns a float32 array shaped (nj, ni) with missing values as np.nan.
    """
    import eccodes

    with open(path, "rb") as f:
        msg = eccodes.codes_grib_new_from_file(f)
        if msg is None:
            raise ValueError("No GRIB2 messages found")
        try:
            nj     = eccodes.codes_get(msg, "Nj")
            ni     = eccodes.codes_get(msg, "Ni")
            values = eccodes.codes_get_values(msg)
            try:
                missing = eccodes.codes_get(msg, "missingValue")
            except Exception:
                missing = 9.999e20
        finally:
            eccodes.codes_release(msg)

    data = values.reshape(nj, ni).astype(np.float32)

    # MRMS fill values: standard GRIB missing AND the MRMS-specific -999.
    data[np.isclose(data, missing, rtol=1e-3)] = np.nan
    data[data <= -900.0] = np.nan   # MRMS uses -999.0 for "no-data"
    return data


def _read_with_cfgrib(path: str) -> np.ndarray:
    """
    Fallback GRIB2 reader using cfgrib / xarray.
    Returns a float32 array shaped (lat, lon) with missing values as np.nan.
    """
    import cfgrib

    datasets = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    for ds in datasets:
        for var in ds.data_vars:
            arr = ds[var].values
            if arr.ndim == 2:
                data = arr.astype(np.float32)
                data[data <= -900.0] = np.nan
                return data
    raise ValueError("No 2-D variable found via cfgrib")


def _read_grib2(path: str) -> np.ndarray:
    """Try eccodes first, fall back to cfgrib."""
    try:
        return _read_with_eccodes(path)
    except Exception as e:
        logger.warning("eccodes read failed (%s), trying cfgrib", e)
        return _read_with_cfgrib(path)


# Public alias so external scripts (e.g. radar_archiver.py) can import this
# without duplicating the eccodes/cfgrib fallback logic.
read_grib2 = _read_grib2


# ── Rendering ─────────────────────────────────────────────────────────────────

def _apply_nws_colormap(data: np.ndarray) -> np.ndarray:
    """
    Map a 2-D reflectivity array (dBZ) to an RGBA uint8 array.
    No-echo (< 5 dBZ) and missing are fully transparent.
    """
    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    for idx, (thresh, r, g, b) in enumerate(_NWS_CMAP):
        upper = _NWS_CMAP[idx + 1][0] if idx + 1 < len(_NWS_CMAP) else 200.0
        mask = (data >= thresh) & (data < upper)
        rgba[mask, 0] = r
        rgba[mask, 1] = g
        rgba[mask, 2] = b
        rgba[mask, 3] = 210   # slight transparency so base map shows through

    # No echo / missing → alpha = 0
    no_echo = (data < 5) | np.isnan(data)
    rgba[no_echo, 3] = 0

    return rgba


def _reproject_to_mercator(data: np.ndarray) -> np.ndarray:
    """
    Reproject a lat/lon-grid array to Mercator-y-linear rows.

    MRMS data has rows equally spaced in latitude (equidistant cylindrical),
    but Leaflet's imageOverlay stretches the image linearly between bounds in
    Web Mercator.  Without reprojection, Leaflet's linear stretch causes
    features to appear displaced northward: at 40°N, the error is ~6% of the
    total map height.

    Input:  (nj, ni) with row 0 = MRMS_BOUNDS["north"], row nj-1 = ["south"]
    Output: (out_h, ni) with rows linearly spaced in Mercator y.
    """
    nj, ni  = data.shape
    lat_n   = MRMS_BOUNDS["north"]   # 54.995°N
    lat_s   = MRMS_BOUNDS["south"]   # 20.005°N

    # Mercator y at each bound
    y_n = np.log(np.tan(np.pi / 4 + np.radians(lat_n) / 2))
    y_s = np.log(np.tan(np.pi / 4 + np.radians(lat_s) / 2))

    # Output height: keep the Mercator aspect ratio so the image isn't
    # artificially squished/stretched when Leaflet displays it.
    lon_range_rad = np.radians(MRMS_BOUNDS["east"] - MRMS_BOUNDS["west"])
    out_h = max(1, int(round(ni * (y_n - y_s) / lon_range_rad)))

    # For each output row, compute the Mercator y → latitude → input row index
    merc_ys = np.linspace(y_n, y_s, out_h)
    lats    = np.degrees(2 * np.arctan(np.exp(merc_ys)) - np.pi / 2)

    row_idx = (lat_n - lats) / (lat_n - lat_s) * (nj - 1)
    row_idx = np.clip(np.round(row_idx).astype(int), 0, nj - 1)

    return data[row_idx, :]


def _render_png(data: np.ndarray, step: int = RENDER_STEP) -> bytes:
    """
    Downsample by RENDER_STEP, reproject to Web Mercator, apply NWS colormap,
    return PNG bytes.
    """
    s = max(1, int(step))
    downsampled = data[::s, ::s]
    reprojected = _reproject_to_mercator(downsampled)
    rgba = _apply_nws_colormap(reprojected)
    img = Image.fromarray(rgba, mode="RGBA")

    buf = io.BytesIO()
    # compress_level=1 is fast; radar PNGs are already sparse so size stays small
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


# NWS precipitation rate color scale (mm/hr).
# Transparent below 0.25 mm/hr ("no precipitation").
_PRECIP_CMAP = [
    (  0.25, 0xa0, 0xf0, 0xa0),   # light green
    (  1.00, 0x00, 0xc8, 0x00),   # green
    (  2.50, 0x00, 0x96, 0x00),   # dark green
    (  5.00, 0xf0, 0xf0, 0x00),   # yellow
    ( 10.00, 0xff, 0xa0, 0x00),   # orange
    ( 25.00, 0xff, 0x00, 0x00),   # red
    ( 50.00, 0xc8, 0x00, 0xc8),   # magenta
    (100.00, 0xff, 0xff, 0xff),   # white (extreme)
]


def apply_precip_colormap(data: np.ndarray) -> np.ndarray:
    """
    Map a 2-D precipitation rate array (mm/hr) to an RGBA uint8 array.
    Values below 0.25 mm/hr and missing are fully transparent.
    """
    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    for idx, (thresh, r, g, b) in enumerate(_PRECIP_CMAP):
        upper = _PRECIP_CMAP[idx + 1][0] if idx + 1 < len(_PRECIP_CMAP) else 1e9
        mask = (data >= thresh) & (data < upper)
        rgba[mask, 0] = r
        rgba[mask, 1] = g
        rgba[mask, 2] = b
        rgba[mask, 3] = 210

    no_precip = (data < 0.25) | np.isnan(data) | (data < 0)
    rgba[no_precip, 3] = 0

    return rgba


def render_precip_png(data: np.ndarray, step: int = RENDER_STEP) -> bytes:
    """
    Downsample by RENDER_STEP, reproject to Web Mercator, apply NWS precip
    rate colormap, return PNG bytes.  Suitable for archiving MRMS PrecipRate frames.
    """
    s = max(1, int(step))
    downsampled = data[::s, ::s]
    reprojected = _reproject_to_mercator(downsampled)
    rgba = apply_precip_colormap(reprojected)
    img = Image.fromarray(rgba, mode="RGBA")

    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


# ── Core fetch pipeline ───────────────────────────────────────────────────────

def _fetch_and_render(key: str) -> dict[int, bytes]:
    """Download, decompress, parse, and render one MRMS key."""
    url = f"{MRMS_S3_BASE}/{key}"
    logger.info("Fetching MRMS radar: %s", url)

    r = requests.get(url, timeout=90)
    r.raise_for_status()

    grib_data = gzip.decompress(r.content)
    logger.info("MRMS decompressed: %.1f MB", len(grib_data) / 1e6)

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(grib_data)
        tmp_path = tmp.name

    try:
        data = _read_grib2(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    pngs: dict[int, bytes] = {}
    for step in RENDER_STEPS:
        pngs[step] = _render_png(data, step=step)
    return pngs


def _select_render_step(requested_step: int | None, available_steps: list[int]) -> int:
    """Pick the closest available render step for the requested detail level."""
    if not available_steps:
        return RENDER_STEP
    if requested_step is None:
        return RENDER_STEP if RENDER_STEP in available_steps else available_steps[0]
    try:
        req = max(1, int(requested_step))
    except (TypeError, ValueError):
        req = RENDER_STEP
    return min(available_steps, key=lambda s: abs(s - req))


def get_cached_composite_png(step: int | None = None) -> tuple[bytes | None, int, datetime | None]:
    """
    Return (png_bytes, used_step, valid_time) from in-memory cache.
    Falls back to default cached PNG if per-step cache is unavailable.
    """
    with _cache_lock:
        png_by_step = _cache.get("png_by_step") or {}
        available = sorted([int(s) for s in png_by_step.keys()], reverse=True)
        used_step = _select_render_step(step, available) if available else RENDER_STEP
        png_bytes = png_by_step.get(used_step) or _cache.get("png_bytes")
        return png_bytes, used_step, _cache.get("valid_time")


# ── Public API ────────────────────────────────────────────────────────────────

def get_radar_composite() -> dict:
    """
    Return a copy of the cache dict:
      { png_bytes, valid_time, fetched_at, error }

    Synchronously fetches fresh data if the cache is stale or empty.
    Callers in the web tier should prefer the background-refresh pattern
    (start_background_refresh) so the HTTP response is never blocked waiting
    for a 60–90 s S3 download.
    """
    with _cache_lock:
        age = time.time() - _cache["fetched_at"]
        if _cache["png_bytes"] is not None and age < CACHE_TTL:
            return dict(_cache)

    key, valid_time = _find_latest_key()
    if key is None:
        with _cache_lock:
            _cache["error"] = "No MRMS files found in S3 bucket"
            return dict(_cache)

    try:
        png_by_step = _fetch_and_render(key)
    except Exception as e:
        logger.error("MRMS fetch/render failed: %s", e, exc_info=True)
        with _cache_lock:
            _cache["error"] = str(e)
            return dict(_cache)   # return stale data if any

    default_png = png_by_step.get(RENDER_STEP) or png_by_step[min(RENDER_STEPS)]
    with _cache_lock:
        _cache["png_bytes"]  = default_png
        _cache["png_by_step"] = png_by_step
        _cache["valid_time"] = valid_time
        _cache["fetched_at"] = time.time()
        _cache["error"]      = None

    logger.info(
        "MRMS radar cached: %.1f KB (step=%s), valid %s",
        len(default_png) / 1024,
        RENDER_STEP,
        valid_time,
    )
    return dict(_cache)


def is_cache_fresh() -> bool:
    """Return True if a valid PNG is cached and not stale."""
    with _cache_lock:
        return (
            _cache["png_bytes"] is not None
            and time.time() - _cache["fetched_at"] < CACHE_TTL
        )


def start_background_refresh() -> None:
    """
    Spawn a daemon thread that pre-fetches and refreshes the radar cache
    every CACHE_TTL seconds.  Call once at app startup.
    """
    def _loop() -> None:
        while True:
            try:
                get_radar_composite()
            except Exception as e:
                logger.error("Background radar refresh error: %s", e)
            time.sleep(CACHE_TTL)

    t = threading.Thread(target=_loop, daemon=True, name="radar-mrms-refresh")
    t.start()
    logger.info("MRMS radar background refresh started (TTL=%ds)", CACHE_TTL)
