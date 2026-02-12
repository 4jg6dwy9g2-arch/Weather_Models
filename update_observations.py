#!/usr/bin/env python3
"""
Update observations for existing forecast runs.

This script goes through all saved forecast runs and re-fetches observations
for valid times that are now in the past. This is useful when:
- Time has passed since the forecast was initially fetched
- More valid times have moved into the past
- You want verification data for longer lead times
"""

import json
from pathlib import Path
from datetime import datetime
import logging
import weatherlink
from app import calculate_all_verification, save_forecasts_db, load_forecasts_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FORECASTS_FILE = Path(__file__).parent / "data" / "forecasts.json"


def update_observations_for_run(location_name: str, run_id: str, run_data: dict) -> bool:
    """
    Update observations for a single forecast run.

    Returns:
        True if observations were updated, False otherwise
    """
    gfs_data = run_data.get("gfs", {})
    forecast_times = gfs_data.get("times", [])

    if not forecast_times:
        logger.warning(f"  No forecast times found for run {run_id}")
        return False

    logger.info(f"  Fetching observations for {len(forecast_times)} forecast times...")

    # Fetch missing WeatherLink data first
    try:
        new_records = weatherlink.fetch_missing_data(silent=False)
        if new_records > 0:
            logger.info(f"  Downloaded {new_records} new WeatherLink records")
    except Exception as e:
        logger.warning(f"  Error fetching missing WeatherLink data: {e}")

    # Get observations for all forecast times
    try:
        observed = weatherlink.get_observations_for_forecast_times(forecast_times)

        # Count how many non-None observations we have
        obs_temps = observed.get("temps", [])
        valid_obs_count = sum(1 for t in obs_temps if t is not None)

        logger.info(f"  Found {valid_obs_count} valid observations out of {len(forecast_times)} times")

        if valid_obs_count == 0:
            logger.warning(f"  No observations available for run {run_id}")
            return False

        # Update the run data with new observations
        run_data["observed"] = observed

        # Recalculate verification metrics
        aifs_data = run_data.get("aifs", {})
        ifs_data = run_data.get("ifs", {})
        verification = calculate_all_verification(gfs_data, aifs_data, observed, ifs_data)

        run_data["verification"] = verification

        logger.info(f"  ✓ Updated observations (temp count: {verification.get('temp_count', 0)})")
        return True

    except Exception as e:
        logger.error(f"  Error getting observations: {e}")
        return False


def main():
    """Update observations for all forecast runs."""
    logger.info("Loading forecasts database...")
    db = load_forecasts_db()

    if not db:
        logger.error("No forecast data found")
        return

    total_updated = 0

    for location_name, loc_data in db.items():
        logger.info(f"\nProcessing location: {location_name}")
        runs = loc_data.get("runs", {})

        if not runs:
            logger.warning(f"  No runs found for {location_name}")
            continue

        logger.info(f"  Found {len(runs)} runs")

        for run_id, run_data in runs.items():
            logger.info(f"\n  Run: {run_id}")

            if update_observations_for_run(location_name, run_id, run_data):
                total_updated += 1

    # Save updated database
    if total_updated > 0:
        logger.info(f"\n✓ Updated {total_updated} runs")
        logger.info("Saving to database...")
        save_forecasts_db(db)
        logger.info(f"✓ Saved to {FORECASTS_FILE}")
    else:
        logger.info("\nNo runs were updated")


if __name__ == "__main__":
    main()
