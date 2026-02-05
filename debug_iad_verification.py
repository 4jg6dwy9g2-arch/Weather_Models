#!/usr/bin/env python3
"""Debug script to trace IAD verification calculation."""

import json
from datetime import datetime, timedelta, timezone

# Load data
with open('asos_forecasts.json') as f:
    db = json.load(f)

station_id = 'IAD'
runs = db.get('runs', {})
observations = db.get('observations', {}).get(station_id, {})

print(f'=== Debugging IAD Verification ===\n')
print(f'Total runs: {len(runs)}')
print(f'IAD observations: {len(observations)}')
print()

# Pick first run
first_run_key = list(runs.keys())[0]
first_run = runs[first_run_key]
init_time = datetime.fromisoformat(first_run_key)
forecast_hours = first_run.get('forecast_hours', [])

print(f'Run: {first_run_key}')
print(f'Forecast hours: {forecast_hours}')
print()

# Check lead time 24
lead_time = 24
if lead_time in forecast_hours:
    idx = forecast_hours.index(lead_time)
    valid_time = init_time + timedelta(hours=lead_time)

    print(f'Lead time {lead_time}h:')
    print(f'  Init: {init_time.isoformat()}')
    print(f'  Valid: {valid_time.isoformat()}')
    print(f'  Index: {idx}')
    print()

    # Get forecast
    gfs_data = first_run.get('gfs', {}).get(station_id, {})
    if gfs_data:
        temps = gfs_data.get('temps', [])
        if idx < len(temps):
            fcst_temp = temps[idx]
            print(f'  Forecast temp: {fcst_temp}')
        else:
            print(f'  Forecast temp: INDEX OUT OF RANGE ({idx} >= {len(temps)})')
    else:
        print(f'  No GFS data for {station_id}')
    print()

    # Look for observation
    print(f'  Looking for observation near {valid_time.isoformat()}...')
    best_match = None
    best_delta = timedelta(minutes=31)

    for obs_time_str, obs_data in observations.items():
        try:
            obs_time = datetime.fromisoformat(obs_time_str)
            delta = abs(obs_time - valid_time)

            if delta < best_delta:
                best_delta = delta
                best_match = (obs_time_str, obs_data)
        except ValueError:
            continue

    if best_match and best_delta <= timedelta(minutes=30):
        obs_time_str, obs_data = best_match
        print(f'  ✓ Found observation at {obs_time_str}')
        print(f'    Delta: {best_delta}')
        print(f'    temp: {obs_data.get("temp")}')
        print(f'    mslp: {obs_data.get("mslp")}')
        print(f'    precip: {obs_data.get("precip")}')

        # Calculate error
        if fcst_temp and obs_data.get('temp'):
            error = fcst_temp - obs_data.get('temp')
            print(f'    Error: {error:.2f}°F')
            print(f'    MAE contribution: {abs(error):.2f}')
    else:
        print(f'  ✗ No observation found within 30 minutes')
        print(f'    Best match: {best_delta} away')

print('\nThis data SHOULD be producing verification metrics!')
