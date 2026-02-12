#!/usr/bin/env python3
"""
Check forecast runs and precipitation verification pairs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asos
from datetime import datetime, timezone, timedelta
from collections import defaultdict

print("=" * 80)
print("CHECKING FORECAST RUNS AND PRECIPITATION VERIFICATION")
print("=" * 80)

db = asos.load_asos_forecasts_db()
runs = db.get("runs", {})
stations = db.get("stations", {})

print(f"\n1. Forecast runs:")
print(f"   Total runs in database: {len(runs)}")

# Check synoptic times
synoptic_runs = []
non_synoptic_runs = []

for run_key in runs.keys():
    try:
        init_time = datetime.fromisoformat(run_key)
        if init_time.hour % 6 == 0:
            synoptic_runs.append(run_key)
        else:
            non_synoptic_runs.append(run_key)
    except:
        pass

print(f"   Synoptic runs (00/06/12/18Z): {len(synoptic_runs)}")
print(f"   Non-synoptic runs: {len(non_synoptic_runs)}")

if synoptic_runs:
    print(f"\n   Recent synoptic runs:")
    for run_key in sorted(synoptic_runs)[-5:]:
        print(f"     - {run_key}")

# Check how many stations have precip forecasts
stations_with_precip_forecast = set()
for run_key, run_data in runs.items():
    gfs_data = run_data.get('gfs', {})
    for station_id, fcst in gfs_data.items():
        if fcst.get('precips'):  # Has precipitation forecast
            stations_with_precip_forecast.add(station_id)

print(f"\n2. Precipitation forecasts:")
print(f"   Stations with precip forecasts: {len(stations_with_precip_forecast)}")
print(f"   Total stations in database: {len(stations)}")

# Count valid precip verification pairs at 6hr lead time
print(f"\n3. Testing verification pairs at 6hr lead time:")

valid_pairs = defaultdict(int)
missing_obs = defaultdict(int)
missing_fcst = defaultdict(int)

now = datetime.now(timezone.utc)

for run_key, run_data in runs.items():
    try:
        init_time = datetime.fromisoformat(run_key)
        if init_time.tzinfo is None:
            init_time = init_time.replace(tzinfo=timezone.utc)
    except:
        continue

    # Check 6hr lead time
    valid_time = init_time + timedelta(hours=6)

    # Skip future times
    if valid_time >= now:
        continue

    # Only check synoptic times (where 6hr precip is calculated)
    if valid_time.hour % 6 != 0:
        continue

    gfs_data = run_data.get('gfs', {})

    for station_id in list(stations.keys())[:100]:  # Sample 100 stations
        fcst = gfs_data.get(station_id, {})
        precip_fcst = fcst.get('precips', [])

        if not precip_fcst or len(precip_fcst) == 0:
            missing_fcst[station_id] += 1
            continue

        # Check if observation exists
        precip_obs = asos.calculate_6hr_precip_total(db, station_id, valid_time)

        if precip_obs is None:
            missing_obs[station_id] += 1
        else:
            valid_pairs[station_id] += 1

print(f"   Sample of 100 stations:")
print(f"   Stations with at least 1 valid pair: {len(valid_pairs)}")
print(f"   Stations with no valid pairs: {100 - len(valid_pairs)}")

if valid_pairs:
    pair_counts = sorted(valid_pairs.values())
    print(f"\n   Valid pairs per station:")
    print(f"     Min: {min(pair_counts)}")
    print(f"     Median: {pair_counts[len(pair_counts)//2]}")
    print(f"     Max: {max(pair_counts)}")

# Check verification cache
print(f"\n4. Verification cache:")
verif = asos.get_verification_data_from_cache('gfs', 'precip', 6)
print(f"   Stations in cache for precip: {len(verif) if verif else 0}")

verif_temp = asos.get_verification_data_from_cache('gfs', 'temp', 6)
print(f"   Stations in cache for temp: {len(verif_temp) if verif_temp else 0}")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
if len(synoptic_runs) < 5:
    print("⚠ Issue: Very few synoptic runs (< 5)")
    print("  Solution: Run more syncs at 00Z, 06Z, 12Z, or 18Z times")
elif len(valid_pairs) < 50:
    print("⚠ Issue: Few valid forecast-observation pairs")
    print("  Possible causes:")
    print("    - Not enough time has passed for forecasts to verify")
    print("    - Forecasts missing for many stations")
else:
    print("✓ Data looks good - cache should have updated")
    print("  Try clearing browser cache and refreshing")
print("=" * 80)
