"""
Compute CONUS 14-day precipitation climatology from existing ERA5 files on T7.
Reads from conus_daily_precip, writes conus_precip_climatology_14day.nc.
No downloads — only processes files already on disk.
"""

import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PRECIP_DIR  = Path("/Volumes/T7/Weather_Models/era5/conus_daily_precip")
OUTPUT_FILE = Path("/Volumes/T7/Weather_Models/era5/conus_precip_climatology_14day.nc")

CLIM_START = 1990
CLIM_END   = 2020

# ── Load all precip files ────────────────────────────────────────────────────

all_files = sorted(glob.glob(str(PRECIP_DIR / "era5_tp_daily_CONUS_*_M*.nc")))
if not all_files:
    sys.exit(f"No files found in {PRECIP_DIR}")

logger.info(f"Found {len(all_files)} files")
logger.info("Opening dataset (lazy, with dask)…")

ds_all = xr.open_mfdataset(all_files, combine='by_coords', parallel=True, engine='netcdf4')

# Filter to climatology period
ds_clim = ds_all.sel(time=slice(f"{CLIM_START}-01-01", f"{CLIM_END}-12-31"))
logger.info(f"Climatology period: {ds_clim.time.values[0]} → {ds_clim.time.values[-1]}")
logger.info(f"Grid: {len(ds_clim.latitude)} lat × {len(ds_clim.longitude)} lon")

lats = ds_clim.latitude.values
lons = ds_clim.longitude.values

# ── Compute climatology ──────────────────────────────────────────────────────

clim_data = np.full((366, len(lats), len(lons)), np.nan, dtype=np.float32)

logger.info("Computing 14-day climatology for each DOY (this takes 10-20 min)…")

for doy in range(1, 367):
    if doy % 30 == 0 or doy == 1:
        logger.info(f"  DOY {doy}/366")

    windows = []
    for year in range(CLIM_START, CLIM_END + 1):
        try:
            if doy == 60:
                # Feb 29 — only exists in leap years
                if not pd.Timestamp(year=year, month=1, day=1).is_leap_year:
                    continue
                start = pd.Timestamp(year=year, month=2, day=29)
            elif doy > 60 and not pd.Timestamp(year=year, month=1, day=1).is_leap_year:
                # Non-leap: DOY 60 = Mar 1, shift by 1
                start = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 2)
            else:
                start = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        except Exception:
            continue

        end = start + pd.Timedelta(days=13)  # 14-day window inclusive
        window = ds_clim['tp'].sel(time=slice(start, end))

        if len(window.time) >= 10:
            windows.append(window.sum(dim='time').values)

    if windows:
        clim_data[doy - 1] = np.nanmean(windows, axis=0)

# ── Write output ─────────────────────────────────────────────────────────────

logger.info(f"Writing → {OUTPUT_FILE}")

ds_out = xr.Dataset(
    {
        'precip_14d_clim': (['doy', 'latitude', 'longitude'], clim_data, {
            'long_name': '14-day Precipitation Climatology',
            'units': 'mm',
            'description': (
                f'Sum of 14-day windows starting on each DOY, '
                f'averaged across {CLIM_START}–{CLIM_END}'
            ),
        })
    },
    coords={
        'doy':       (['doy'],       np.arange(1, 367),  {'long_name': 'Day of Year'}),
        'latitude':  (['latitude'],  lats,               {'units': 'degrees_north'}),
        'longitude': (['longitude'], lons,               {'units': 'degrees_east'}),
    },
    attrs={
        'title': 'CONUS Precipitation 14-day Climatology',
        'climatology_period': f'{CLIM_START}-{CLIM_END}',
        'created': pd.Timestamp.now().isoformat(),
    }
)

ds_out.to_netcdf(OUTPUT_FILE, encoding={
    'precip_14d_clim': {'dtype': 'float32', 'zlib': True, 'complevel': 5}
})

mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
logger.info(f"Done. File size: {mb:.1f} MB")
logger.info(f"Output: {OUTPUT_FILE}")

ds_all.close()
