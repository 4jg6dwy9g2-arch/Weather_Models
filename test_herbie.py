from datetime import datetime, timedelta, timezone
from herbie import Herbie
import xarray as xr
import numpy as np
import tempfile
from pathlib import Path

# Define the Fairfax, VA region (approximate bounds)
# Coordinates: 38.8462° N, 77.3064° W
# Using a slightly larger box around Fairfax to ensure coverage
FAIRFAX_WEST_NEG180 = -77.5
FAIRFAX_EAST_NEG180 = -77.1
FAIRFAX_SOUTH = 38.7
FAIRFAX_NORTH = 39.0

# Convert longitude from -180 to 180 to 0 to 360 range
FAIRFAX_WEST_0_360 = FAIRFAX_WEST_NEG180 + 360
FAIRFAX_EAST_0_360 = FAIRFAX_EAST_NEG180 + 360

# Latest GFS run (adjust as needed for testing current data)
# Using a fixed date for reproducibility - MAKE SURE THIS IS A RECENT DATE
init_time = (datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=6)).replace(tzinfo=None) # Use a recent init time, e.g., 6 hours ago
if init_time.hour not in [0, 6, 12, 18]: # GFS init hours
    while init_time.hour not in [0, 6, 12, 18]:
        init_time -= timedelta(hours=1)

forecast_hour = 0 # F000 for initial test

print(f"Attempting to fetch GFS data for {init_time} F{forecast_hour:03d}")

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_save_dir = Path(tmpdir)
    # Initialize Herbie
    H = Herbie(
        init_time,
        model="gfs",
        product="pgrb2.0p25",  # 0.25 degree resolution
        fxx=forecast_hour,
        save_dir=tmp_save_dir,
        verbose=True # Enable verbose output for Herbie
    )

    # Test 2m Temperature
    try:
        print("\nFetching 2m Temperature...")
        ds_temp = H.xarray(":TMP:2 m above ground:", remove_grib=True)
        
        if 't2m' in ds_temp.data_vars:
            temp_da = ds_temp.t2m
        else:
            # Try to find the temperature variable if not named 't2m'
            temp_da = None
            for var_name in ds_temp.data_vars:
                if 'tmp' in var_name.lower() and '2m' in var_name.lower():
                    temp_da = ds_temp[var_name]
                    break
            if temp_da is None:
                raise ValueError("Could not find 2m Temperature variable in dataset.")

        print(f"Temperature DataArray coordinates: {temp_da.coords}")
        print(f"Temperature DataArray dims: {temp_da.dims}")

        # Subset to region
        subset_temp = temp_da.sel(
            latitude=slice(FAIRFAX_SOUTH, FAIRFAX_NORTH),
            longitude=slice(FAIRFAX_WEST_0_360, FAIRFAX_EAST_0_360)
        )
        
        print(f"Subset Temperature data shape: {subset_temp.shape}")
        print(f"Subset Temperature data values (mean): {subset_temp.values.mean()}")

    except Exception as e:
        print(f"Error fetching 2m Temperature: {e}")

    # Test Total Precipitation
    try:
        print("\nFetching Total Precipitation (Accumulated)...")
        # A common Herbie search string for GFS accumulated precipitation.
        # Note: GFS accumulated precipitation is often ':APCP:surface:anl', ':APCP:surface:6 hour acc fcst', etc.
        # For F000, it's typically an accumulated value from the previous run or 0.
        # We will search for a generic accumulated precipitation.
        
        ds_precip = H.xarray(":APCP:", remove_grib=True) # or try ":APCP:surface:anl"
        
        precip_da = None
        # Common GFS variable names for accumulated precipitation
        possible_precip_vars = ['tp', 'prate'] 
        for var_name in ds_precip.data_vars:
            if var_name in possible_precip_vars:
                precip_da = ds_precip[var_name]
                break

        if precip_da is None:
            # Fallback to general search if specific names not found
            for var_name in ds_precip.data_vars:
                if 'precip' in var_name.lower() or 'apcp' in var_name.lower():
                    precip_da = ds_precip[var_name]
                    break
        
        if precip_da is None:
            raise ValueError("Could not find precipitation variable in dataset.")

        print(f"Precipitation DataArray coordinates: {precip_da.coords}")
        print(f"Precipitation DataArray dims: {precip_da.dims}")

        # Subset to region
        subset_precip = precip_da.sel(
            latitude=slice(FAIRFAX_SOUTH, FAIRFAX_NORTH),
            longitude=slice(FAIRFAX_WEST_0_360, FAIRFAX_EAST_0_360)
        )

        print(f"Subset Precipitation data shape: {subset_precip.shape}")
        print(f"Subset Precipitation data values (mean): {subset_precip.values.mean()}")

    except Exception as e:
        print(f"Error fetching Total Precipitation: {e}")

