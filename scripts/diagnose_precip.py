#!/usr/bin/env python3
"""
Diagnostic script to understand why precipitation verification has low coverage.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asos
from datetime import datetime, timezone, timedelta

print("=" * 80)
print("PRECIPITATION VERIFICATION DIAGNOSTIC")
print("=" * 80)

# Load the ASOS database
print("\n1. Loading ASOS cache...")
db = asos.load_asos_forecasts_db()

observations = db.get("observations", {})
print(f"   Total stations in database: {len(observations)}")

# Check how many stations have ANY precipitation data
stations_with_precip = 0
stations_with_recent_precip = 0
total_precip_obs = 0

cutoff_time = datetime.now(timezone.utc) - timedelta(days=30)

for station_id, obs_data in observations.items():
    has_precip = False
    has_recent_precip = False

    for obs_time_str, obs in obs_data.items():
        if obs.get('precip') is not None:
            has_precip = True
            total_precip_obs += 1

            try:
                obs_time = datetime.fromisoformat(obs_time_str)
                if obs_time > cutoff_time:
                    has_recent_precip = True
            except:
                pass

    if has_precip:
        stations_with_precip += 1
    if has_recent_precip:
        stations_with_recent_precip += 1

print(f"   Stations with ANY precip observations: {stations_with_precip}")
print(f"   Stations with precip in last 30 days: {stations_with_recent_precip}")
print(f"   Total precip observations: {total_precip_obs}")

# Check verification data
print("\n2. Checking verification cache...")
verif_precip = asos.get_verification_data_from_cache('gfs', 'precip', 6)
verif_temp = asos.get_verification_data_from_cache('gfs', 'temp', 6)

print(f"   GFS Precip verification at 6hr: {len(verif_precip) if verif_precip else 0} stations")
print(f"   GFS Temp verification at 6hr: {len(verif_temp) if verif_temp else 0} stations")

# Test 6-hour precipitation calculation for a sample of stations
print("\n3. Testing 6-hour precip calculation for sample stations...")
print("   (Checking if stations have enough hourly data for 6hr accumulation)")

# Get a recent synoptic time
now = datetime.now(timezone.utc)
synoptic_hour = (now.hour // 6) * 6
test_time = now.replace(hour=synoptic_hour, minute=0, second=0, microsecond=0)
if test_time > now:
    test_time -= timedelta(hours=6)

print(f"   Testing for time: {test_time.isoformat()}")

success_count = 0
fail_insufficient = 0
fail_no_data = 0

# Test a sample of 100 stations
sample_stations = list(observations.keys())[:100]

for station_id in sample_stations:
    result = asos.calculate_6hr_precip_total(db, station_id, test_time)

    if result is not None:
        success_count += 1
    else:
        # Check why it failed
        station_obs = observations.get(station_id, {})
        if not station_obs:
            fail_no_data += 1
        else:
            # Has some data but not enough for 6hr
            fail_insufficient += 1

print(f"   Sample results (100 stations):")
print(f"     - Successful 6hr calculation: {success_count}")
print(f"     - Failed (insufficient hourly data): {fail_insufficient}")
print(f"     - Failed (no observations): {fail_no_data}")
print(f"   Success rate: {success_count}%")

# Extrapolate to full network
estimated_total = int((success_count / 100) * len(observations))
print(f"\n   Estimated total stations with valid 6hr precip: ~{estimated_total}")
print(f"   Actual verification showing: {len(verif_precip) if verif_precip else 0}")

print("\n4. Checking observation timing...")
# Check if observations are at the right times for 6hr accumulation
recent_obs_times = {}
for station_id in list(observations.keys())[:50]:
    station_obs = observations.get(station_id, {})
    for obs_time_str in list(station_obs.keys())[-10:]:  # Last 10 obs
        try:
            obs_time = datetime.fromisoformat(obs_time_str)
            hour = obs_time.hour
            recent_obs_times[hour] = recent_obs_times.get(hour, 0) + 1
        except:
            pass

print("   Recent observation times (sample of 50 stations, last 10 obs each):")
for hour in sorted(recent_obs_times.keys()):
    print(f"     Hour {hour:02d}Z: {recent_obs_times[hour]} observations")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
