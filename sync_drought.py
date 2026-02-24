#!/usr/bin/env python3
"""Standalone script to refresh the US Drought Monitor cache and
update the drought-crop exposure time series with the latest week's data.

Run via LaunchAgent every Friday at midnight:
  ~/Library/LaunchAgents/com.weathermodels.droughtmonitor.plist
"""
import sys
import drought_monitor
import drought_crops

# 1. Refresh the USDM GeoJSON cache
result = drought_monitor.sync_drought_monitor()
print(result)

# 2. Append the latest Tuesday to the crop-exposure time series.
#    build_timeseries() skips dates already in cache, so this only
#    fetches and computes the one new week.
crop_result = drought_crops.build_timeseries(
    progress_cb=lambda msg: print(msg, flush=True)
)
print(f"Drought-crop cache: {len(crop_result['dates'])} weeks total.")

sys.exit(0 if result.get("success") else 1)
