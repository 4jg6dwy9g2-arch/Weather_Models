#!/usr/bin/env python3
"""Rebuild the drought-crop exposure time series cache.

Run once (takes 1–3 hours for the full 2024 backfill) or weekly to append
the latest Tuesday's data.

Usage:
    python compute_drought_crops.py
    python compute_drought_crops.py --start 2024-06-01
"""
import argparse
import drought_crops

parser = argparse.ArgumentParser(
    description="Compute weekly crop-drought exposure time series."
)
parser.add_argument(
    "--start",
    default="2024-01-02",
    metavar="YYYY-MM-DD",
    help="First Tuesday to include (default: 2024-01-02)",
)
args = parser.parse_args()

result = drought_crops.build_timeseries(
    start_date=args.start,
    progress_cb=lambda msg: print(msg, flush=True),
)
print(f"\nDone. {len(result['dates'])} weeks in cache.")
