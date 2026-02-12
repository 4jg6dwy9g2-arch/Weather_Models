#!/usr/bin/env python3
"""
Test the new 6-hour precipitation calculation approach.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asos
from datetime import datetime, timezone, timedelta

print("=" * 80)
print("TESTING NEW 6-HOUR PRECIPITATION CALCULATION")
print("=" * 80)

# Load database
print("\n1. Loading ASOS database...")
db = asos.load_asos_forecasts_db()
observations = db.get("observations", {})
print(f"   Total stations: {len(observations)}")

# Test on a recent synoptic time
now = datetime.now(timezone.utc)
synoptic_hour = (now.hour // 6) * 6
test_time = now.replace(hour=synoptic_hour, minute=0, second=0, microsecond=0)
if test_time > now:
    test_time -= timedelta(hours=6)

print(f"\n2. Testing 6-hour precip calculation at {test_time.isoformat()}")
print("   (6-hour window ending at this time)")

# Test on a sample of stations
sample_size = 200
sample_stations = list(observations.keys())[:sample_size]

success_count = 0
fail_count = 0
total_precip = 0

for station_id in sample_stations:
    result = asos.calculate_6hr_precip_total(db, station_id, test_time)

    if result is not None:
        success_count += 1
        total_precip += result
    else:
        fail_count += 1

print(f"\n3. Results from {sample_size} station sample:")
print(f"   ✓ Successful calculations: {success_count} ({success_count/sample_size*100:.1f}%)")
print(f"   ✗ Failed (no data): {fail_count} ({fail_count/sample_size*100:.1f}%)")
print(f"   Total precip across sample: {total_precip:.2f} inches")

# Extrapolate to full network
estimated_coverage = int((success_count / sample_size) * len(observations))
print(f"\n4. Estimated coverage:")
print(f"   ~{estimated_coverage} out of {len(observations)} stations")
print(f"   Coverage rate: {estimated_coverage/len(observations)*100:.1f}%")

# Compare to current verification cache
print(f"\n5. Current verification cache:")
verif = asos.get_verification_data_from_cache('gfs', 'precip', 6)
print(f"   Stations in cache: {len(verif) if verif else 0}")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
if success_count > 150:  # 75% success rate
    print("✓ New approach working well! Rebuild verification cache to apply.")
    print("  Run: python3 -c 'import asos; asos.precompute_verification_cache()'")
else:
    print("⚠ Success rate still low. May need further investigation.")
print("=" * 80)
