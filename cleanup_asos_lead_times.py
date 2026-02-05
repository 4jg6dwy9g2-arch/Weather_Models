#!/usr/bin/env python3
"""
Cleanup script to reduce ASOS forecast lead times.
Keeps 6-hour intervals for F000-F024, then 24-hour intervals after that.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ASOS_FORECASTS_FILE = Path(__file__).parent / "asos_forecasts.json"

def filter_lead_times(forecast_hours):
    """
    Filter lead times to keep:
    - 6-hour intervals for 0-24 hours: [6, 12, 18, 24]
    - 24-hour intervals after 24 hours: [48, 72, 96, 120, ...]

    Args:
        forecast_hours: List of all forecast hours

    Returns:
        Tuple of (filtered_hours, indices_to_keep)
    """
    filtered = []
    indices = []

    for i, hour in enumerate(forecast_hours):
        # Keep 6-hour intervals up to 24 hours
        if hour <= 24:
            if hour % 6 == 0:
                filtered.append(hour)
                indices.append(i)
        # Keep 24-hour intervals after 24 hours
        elif hour % 24 == 0:
            filtered.append(hour)
            indices.append(i)

    return filtered, indices


def filter_array(arr, indices):
    """Filter an array to keep only specified indices."""
    if not arr or not isinstance(arr, list):
        return arr
    return [arr[i] if i < len(arr) else None for i in indices]


def cleanup_asos_data():
    """Clean up ASOS forecasts to use reduced lead times."""

    if not ASOS_FORECASTS_FILE.exists():
        logger.error(f"File not found: {ASOS_FORECASTS_FILE}")
        return

    logger.info(f"Loading ASOS forecasts from {ASOS_FORECASTS_FILE}...")
    with open(ASOS_FORECASTS_FILE) as f:
        data = json.load(f)

    original_size = ASOS_FORECASTS_FILE.stat().st_size / (1024**3)
    logger.info(f"Original file size: {original_size:.2f} GB")

    runs = data.get("runs", {})
    logger.info(f"Processing {len(runs)} forecast runs...")

    total_filtered = 0

    for run_key, run_data in runs.items():
        original_hours = run_data.get("forecast_hours", [])
        if not original_hours:
            continue

        # Filter lead times
        filtered_hours, keep_indices = filter_lead_times(original_hours)

        logger.info(f"Run {run_key}: {len(original_hours)} -> {len(filtered_hours)} hours")
        logger.debug(f"  Original: {original_hours}")
        logger.debug(f"  Filtered: {filtered_hours}")

        # Update forecast hours
        run_data["forecast_hours"] = filtered_hours

        # Filter forecast data for each model
        for model in ['gfs', 'aifs', 'ifs']:
            model_data = run_data.get(model)
            if not model_data:
                continue

            # Filter each station's forecast arrays
            for station_id, fcst_data in model_data.items():
                if 'temps' in fcst_data:
                    fcst_data['temps'] = filter_array(fcst_data['temps'], keep_indices)
                if 'mslps' in fcst_data:
                    fcst_data['mslps'] = filter_array(fcst_data['mslps'], keep_indices)
                if 'precips' in fcst_data:
                    fcst_data['precips'] = filter_array(fcst_data['precips'], keep_indices)

        total_filtered += len(original_hours) - len(filtered_hours)

    logger.info(f"Filtered out {total_filtered} forecast hours across all runs")

    # Clear cumulative stats since lead times changed
    logger.info("Clearing cumulative stats (will be rebuilt on next sync)...")
    data["cumulative_stats"] = {
        "by_station": {},
        "by_lead_time": {},
        "time_series": {}
    }

    # Save cleaned data
    logger.info("Saving cleaned data...")
    with open(ASOS_FORECASTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    new_size = ASOS_FORECASTS_FILE.stat().st_size / (1024**3)
    logger.info(f"New file size: {new_size:.2f} GB (saved {original_size - new_size:.2f} GB)")
    logger.info("Cleanup complete!")

    # Now regenerate the cache
    logger.info("Regenerating verification cache...")
    try:
        import asos
        asos.precompute_verification_cache()
        logger.info("Cache regeneration complete!")
    except Exception as e:
        logger.error(f"Error regenerating cache: {e}")
        logger.info("You can manually regenerate it later by running:")
        logger.info("  python3 -c 'import asos; asos.precompute_verification_cache()'")


if __name__ == "__main__":
    cleanup_asos_data()
