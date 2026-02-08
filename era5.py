"""
ERA5 Reanalysis Data Fetcher

Fetches historical reanalysis data from the Copernicus Climate Data Store (CDS).
ERA5 provides hourly data from 1940-present at 0.25° resolution.

Setup Instructions:
1. Register for a free account at: https://cds.climate.copernicus.eu/user/register
2. Install the CDS API: pip install cdsapi
3. Get your API credentials from: https://cds.climate.copernicus.eu/api-how-to
4. Create ~/.cdsapirc with:
   url: https://cds.climate.copernicus.eu/api/v2
   key: {uid}:{api-key}

Documentation: https://confluence.ecmwf.int/display/CKB/ERA5
"""

import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import zipfile
import os

logger = logging.getLogger(__name__)

# Fairfax Station location
DEFAULT_LAT = 38.8419
DEFAULT_LON = -77.3091

# Data cache directory (T7 external drive)
CACHE_DIR = Path("/Volumes/T7/Weather_Models/era5")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_era5_surface(
    start_date,
    end_date,
    lat=DEFAULT_LAT,
    lon=DEFAULT_LON,
    variables=None,
    area_buffer=0.5
):
    """
    Fetch ERA5 surface (single-level) data.

    Args:
        start_date: Start date (datetime or 'YYYY-MM-DD')
        end_date: End date (datetime or 'YYYY-MM-DD')
        lat: Latitude
        lon: Longitude
        variables: List of variable names (or None for default set)
        area_buffer: Degrees to expand bounding box around point

    Returns:
        xarray.Dataset with requested variables
    """
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date)

    # Default surface variables
    if variables is None:
        variables = [
            '2m_temperature',           # 2-meter temperature
            '2m_dewpoint_temperature',  # 2-meter dewpoint
            'surface_pressure',          # Surface pressure
            'mean_sea_level_pressure',   # MSLP
            'total_precipitation',       # Precipitation
            '10m_u_component_of_wind',   # U wind component
            '10m_v_component_of_wind',   # V wind component
            'surface_solar_radiation_downwards',  # Solar radiation
            'total_cloud_cover',         # Cloud cover
        ]

    # Bounding box [North, West, South, East]
    area = [
        lat + area_buffer,
        lon - area_buffer,
        lat - area_buffer,
        lon + area_buffer
    ]

    # Generate actual date range (only dates we need)
    date_range = pd.date_range(start_date, end_date, freq='D')
    years = sorted(set(str(d.year) for d in date_range))
    months = sorted(set(f"{d.month:02d}" for d in date_range))
    days = sorted(set(f"{d.day:02d}" for d in date_range))

    # Build cache filename
    cache_file = CACHE_DIR / f"era5_surface_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{lat:.4f}_{lon:.4f}.nc"

    if cache_file.exists():
        logger.info(f"Loading cached ERA5 data: {cache_file}")
        return xr.open_dataset(cache_file)

    logger.info(f"Fetching ERA5 surface data from {start_date} to {end_date}")

    # Initialize CDS API client
    c = cdsapi.Client()

    # Download data
    temp_file = cache_file.with_suffix('.nc.tmp')
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': variables,
            'year': [str(y) for y in years],
            'month': months,
            'day': days,
            'time': [f"{h:02d}:00" for h in range(24)],  # All hours
            'area': area,
            'format': 'netcdf',
        },
        str(temp_file)
    )

    # Check if downloaded file is a ZIP and extract if needed
    nc_file = temp_file
    if zipfile.is_zipfile(temp_file):
        logger.info("Extracting NetCDF from ZIP file...")
        with zipfile.ZipFile(temp_file, 'r') as zip_ref:
            # Extract first .nc file found
            nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
            if nc_files:
                zip_ref.extract(nc_files[0], temp_file.parent)
                nc_file = temp_file.parent / nc_files[0]
                logger.info(f"Extracted: {nc_files[0]}")

    # Load and process data
    ds = xr.open_dataset(nc_file)

    # Rename valid_time to time if present (ERA5 format)
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})

    # Filter to exact date range
    ds = ds.sel(time=slice(start_date, end_date))

    # Save to cache
    ds.to_netcdf(cache_file)

    # Clean up temp files
    if temp_file.exists():
        temp_file.unlink()
    if nc_file != temp_file and nc_file.exists():
        nc_file.unlink()

    logger.info(f"ERA5 data cached to: {cache_file}")
    return ds


def get_era5_pressure_levels(
    start_date,
    end_date,
    lat=DEFAULT_LAT,
    lon=DEFAULT_LON,
    levels=None,
    variables=None,
    area_buffer=0.5
):
    """
    Fetch ERA5 pressure-level data (upper air).

    Args:
        start_date: Start date (datetime or 'YYYY-MM-DD')
        end_date: End date (datetime or 'YYYY-MM-DD')
        lat: Latitude
        lon: Longitude
        levels: List of pressure levels in hPa (or None for default)
        variables: List of variable names (or None for default set)
        area_buffer: Degrees to expand bounding box around point

    Returns:
        xarray.Dataset with requested variables
    """
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date)

    # Default pressure levels (hPa)
    if levels is None:
        levels = [1000, 925, 850, 700, 500, 300, 250, 200]

    # Default upper-air variables
    if variables is None:
        variables = [
            'geopotential',        # Geopotential height
            'temperature',         # Temperature
            'u_component_of_wind', # U wind
            'v_component_of_wind', # V wind
            'relative_humidity',   # RH
        ]

    # Bounding box [North, West, South, East]
    area = [
        lat + area_buffer,
        lon - area_buffer,
        lat - area_buffer,
        lon + area_buffer
    ]

    # Generate actual date range (only dates we need)
    date_range = pd.date_range(start_date, end_date, freq='D')
    years = sorted(set(str(d.year) for d in date_range))
    months = sorted(set(f"{d.month:02d}" for d in date_range))
    days = sorted(set(f"{d.day:02d}" for d in date_range))

    # Build cache filename
    cache_file = CACHE_DIR / f"era5_pressure_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{lat:.4f}_{lon:.4f}.nc"

    if cache_file.exists():
        logger.info(f"Loading cached ERA5 pressure data: {cache_file}")
        return xr.open_dataset(cache_file)

    logger.info(f"Fetching ERA5 pressure-level data from {start_date} to {end_date}")

    # Initialize CDS API client
    c = cdsapi.Client()

    # Download data
    temp_file = cache_file.with_suffix('.nc.tmp')
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': variables,
            'pressure_level': [str(lev) for lev in levels],
            'year': [str(y) for y in years],
            'month': months,
            'day': days,
            'time': [f"{h:02d}:00" for h in range(24)],  # All hours
            'area': area,
            'format': 'netcdf',
        },
        str(temp_file)
    )

    # Check if downloaded file is a ZIP and extract if needed
    nc_file = temp_file
    if zipfile.is_zipfile(temp_file):
        logger.info("Extracting NetCDF from ZIP file...")
        with zipfile.ZipFile(temp_file, 'r') as zip_ref:
            # Extract first .nc file found
            nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
            if nc_files:
                zip_ref.extract(nc_files[0], temp_file.parent)
                nc_file = temp_file.parent / nc_files[0]
                logger.info(f"Extracted: {nc_files[0]}")

    # Load and process data
    ds = xr.open_dataset(nc_file)

    # Rename valid_time to time if present (ERA5 format)
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})

    # Filter to exact date range
    ds = ds.sel(time=slice(start_date, end_date))

    # Convert geopotential to geopotential height (divide by 9.80665 m/s²)
    if 'z' in ds:
        ds['z'] = ds['z'] / 9.80665
        ds['z'].attrs['units'] = 'm'
        ds['z'].attrs['long_name'] = 'Geopotential Height'

    # Save to cache
    ds.to_netcdf(cache_file)

    # Clean up temp files
    if temp_file.exists():
        temp_file.unlink()
    if nc_file != temp_file and nc_file.exists():
        nc_file.unlink()

    logger.info(f"ERA5 pressure data cached to: {cache_file}")
    return ds


def extract_point_timeseries(ds, lat, lon):
    """
    Extract time series for a single point from gridded data.

    Args:
        ds: xarray.Dataset
        lat: Target latitude
        lon: Target longitude

    Returns:
        xarray.Dataset with nearest grid point
    """
    # Find nearest grid point
    point_ds = ds.sel(latitude=lat, longitude=lon, method='nearest')
    return point_ds


def era5_to_dataframe(ds):
    """
    Convert ERA5 xarray Dataset to pandas DataFrame.

    Args:
        ds: xarray.Dataset (should be point data, not gridded)

    Returns:
        pandas.DataFrame with time index
    """
    df = ds.to_dataframe()

    # Convert temperature from Kelvin to Fahrenheit if present
    temp_vars = ['t2m', 't', 'd2m']
    for var in temp_vars:
        if var in df.columns:
            df[var] = (df[var] - 273.15) * 9/5 + 32

    # Convert precipitation from meters to inches if present
    if 'tp' in df.columns:
        df['tp'] = df['tp'] * 39.3701

    # Convert pressure from Pa to mb if present
    if 'sp' in df.columns:
        df['sp'] = df['sp'] / 100
    if 'msl' in df.columns:
        df['msl'] = df['msl'] / 100

    # Calculate wind speed from components
    if 'u10' in df.columns and 'v10' in df.columns:
        df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2) * 2.23694  # m/s to mph
        df['wind_dir'] = (270 - np.arctan2(df['v10'], df['u10']) * 180/np.pi) % 360

    return df


def get_daily_summary(start_date, end_date, lat=DEFAULT_LAT, lon=DEFAULT_LON):
    """
    Get daily summary statistics from ERA5 data.

    Args:
        start_date: Start date
        end_date: End date
        lat: Latitude
        lon: Longitude

    Returns:
        pandas.DataFrame with daily statistics
    """
    # Fetch surface data
    ds = get_era5_surface(start_date, end_date, lat, lon)

    # Extract point data
    point_ds = extract_point_timeseries(ds, lat, lon)

    # Convert to DataFrame
    df = era5_to_dataframe(point_ds)

    # Calculate daily statistics
    daily = pd.DataFrame()

    if 't2m' in df.columns:
        daily['temp_max'] = df['t2m'].resample('D').max()
        daily['temp_min'] = df['t2m'].resample('D').min()
        daily['temp_mean'] = df['t2m'].resample('D').mean()

    if 'd2m' in df.columns:
        daily['dewpoint_mean'] = df['d2m'].resample('D').mean()

    if 'tp' in df.columns:
        # Total precipitation is cumulative, need to diff
        precip = df['tp'].diff()
        precip[precip < 0] = df['tp'][precip < 0]  # Handle resets
        daily['precip_total'] = precip.resample('D').sum()

    if 'msl' in df.columns:
        daily['pressure_mean'] = df['msl'].resample('D').mean()

    if 'wind_speed' in df.columns:
        daily['wind_speed_mean'] = df['wind_speed'].resample('D').mean()
        daily['wind_speed_max'] = df['wind_speed'].resample('D').max()

    if 'ssrd' in df.columns:
        # Solar radiation (J/m²) - convert to average W/m² for daylight hours
        daily['solar_rad_mean'] = df['ssrd'].resample('D').sum() / 86400

    return daily


def get_500mb_history(start_date, end_date, lat=DEFAULT_LAT, lon=DEFAULT_LON):
    """
    Get 500 hPa geopotential height history for Rossby wave analysis.

    Args:
        start_date: Start date
        end_date: End date
        lat: Latitude
        lon: Longitude

    Returns:
        pandas.Series with 500mb heights
    """
    # Fetch 500mb data only
    ds = get_era5_pressure_levels(
        start_date, end_date, lat, lon,
        levels=[500],
        variables=['geopotential']
    )

    # Extract point data
    point_ds = extract_point_timeseries(ds, lat, lon)

    # Get 500mb heights
    if 'z' in point_ds:
        heights = point_ds['z'].sel(level=500).to_series()
        return heights

    return None


def get_climatology(start_year, end_year, lat=DEFAULT_LAT, lon=DEFAULT_LON):
    """
    Calculate climatology (day-of-year averages) for a multi-year period.

    Args:
        start_year: Start year (e.g., 1991)
        end_year: End year (e.g., 2020)
        lat: Latitude
        lon: Longitude

    Returns:
        pandas.DataFrame with climatology by day of year
    """
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    # Get daily data
    daily = get_daily_summary(start_date, end_date, lat, lon)

    # Add day of year
    daily['doy'] = daily.index.dayofyear

    # Calculate statistics by day of year
    climo = daily.groupby('doy').agg({
        'temp_max': ['mean', 'std', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        'temp_min': ['mean', 'std', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        'precip_total': ['mean', 'sum'],
    })

    return climo


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example 1: Get last 30 days of surface data
    print("Example 1: Fetching recent surface data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    try:
        ds = get_era5_surface(start_date, end_date)
        print(f"Fetched {len(ds.time)} time steps")
        print(f"Variables: {list(ds.data_vars)}")

        # Extract point data
        point_ds = extract_point_timeseries(ds, DEFAULT_LAT, DEFAULT_LON)
        df = era5_to_dataframe(point_ds)
        print(f"\nPoint data shape: {df.shape}")
        print(df.head())

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Registered at https://cds.climate.copernicus.eu")
        print("2. Installed cdsapi: pip install cdsapi")
        print("3. Created ~/.cdsapirc with your credentials")

    # Example 2: Get daily summary
    print("\n\nExample 2: Daily summary...")
    try:
        daily = get_daily_summary(start_date, end_date)
        print(daily.head())
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Get 500mb history for wave analysis
    print("\n\nExample 3: 500mb height history...")
    try:
        heights = get_500mb_history(start_date, end_date)
        print(heights.head())
    except Exception as e:
        print(f"Error: {e}")
