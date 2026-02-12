"""
ERA5 Bulk Download Script for Global 500mb Geopotential Heights
and CONUS Daily Precipitation
Optimized for downloading large historical datasets (1940-present)
"""

import cdsapi
import xarray as xr
from pathlib import Path
from datetime import datetime
import logging
import zipfile
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cache directories on T7 drive
CACHE_DIR = Path("/Volumes/T7/Weather_Models/era5/global_500mb")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR_DAILY_PRECIP = Path("/Volumes/T7/Weather_Models/era5/conus_daily_precip")
CACHE_DIR_DAILY_PRECIP.mkdir(parents=True, exist_ok=True)
CACHE_DIR_DAILY_TEMP = Path("/Volumes/T7/Weather_Models/era5/conus_daily_temp")
CACHE_DIR_DAILY_TEMP.mkdir(parents=True, exist_ok=True)
CACHE_DIR_CLIMATOLOGY = Path("/Volumes/T7/Weather_Models/era5")
CACHE_DIR_CLIMATOLOGY.mkdir(parents=True, exist_ok=True)

# CONUS bounding box (North, West, South, East)
CONUS_AREA = [49, -125, 24, -66]


def download_global_500mb_chunk(
    start_year,
    end_year,
    spatial_resolution='1.0/1.0',  # 1° resolution (vs 0.25° default)
    temporal_frequency='daily',      # 'hourly', '12hourly', or 'daily'
    hemisphere='northern'            # 'northern', 'southern', or 'global'
):
    """
    Download a multi-year chunk of global 500mb geopotential heights.

    Args:
        start_year: Start year (e.g., 1940)
        end_year: End year (e.g., 1949)
        spatial_resolution: Grid resolution ('1.0/1.0' = 1°, '0.25/0.25' = 0.25°)
        temporal_frequency: 'hourly', '12hourly' (00Z,12Z), or 'daily' (00Z only)
        hemisphere: 'northern' (20N-90N), 'southern' (90S-20S), or 'global'

    Returns:
        Path to cached NetCDF file
    """

    # Define spatial area
    if hemisphere == 'northern':
        area = [90, -180, 20, 180]  # North, West, South, East
        area_name = 'NH'
    elif hemisphere == 'southern':
        area = [-20, -180, -90, 180]
        area_name = 'SH'
    else:  # global
        area = [90, -180, -90, 180]
        area_name = 'GLOBAL'

    # Define time sampling
    if temporal_frequency == 'hourly':
        times = [f"{h:02d}:00" for h in range(24)]
        freq_name = 'hourly'
    elif temporal_frequency == '12hourly':
        times = ['00:00', '12:00']
        freq_name = '12hr'
    else:  # daily
        times = ['00:00']
        freq_name = 'daily'

    # Build cache filename
    res_name = spatial_resolution.replace('/', 'x').replace('.', 'p')
    cache_file = CACHE_DIR / f"era5_z500_{area_name}_{start_year}-{end_year}_{freq_name}_{res_name}.nc"

    if cache_file.exists():
        logger.info(f"Using cached file: {cache_file}")
    return cache_file

    logger.info(f"Downloading ERA5 500mb heights: {start_year}-{end_year}")
    logger.info(f"  Area: {hemisphere} hemisphere")
    logger.info(f"  Resolution: {spatial_resolution}")
    logger.info(f"  Frequency: {temporal_frequency}")
    logger.info(f"  Estimated size: {_estimate_size(start_year, end_year, spatial_resolution, temporal_frequency, hemisphere)}")

    # Year list
    years = [str(y) for y in range(start_year, end_year + 1)]

    # Initialize CDS API
    c = cdsapi.Client()

    # Download
    temp_file = cache_file.with_suffix('.nc.tmp')

    try:
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': 'geopotential',
                'pressure_level': '500',
                'year': years,
                'month': [f"{m:02d}" for m in range(1, 13)],
                'day': [f"{d:02d}" for d in range(1, 32)],
                'time': times,
                'area': area,
                'grid': [spatial_resolution.split('/')[0], spatial_resolution.split('/')[1]],
                'format': 'netcdf',
            },
            str(temp_file)
        )

        # Extract from ZIP if needed
        nc_file = temp_file
        if zipfile.is_zipfile(temp_file):
            logger.info("Extracting NetCDF from ZIP...")
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                if nc_files:
                    zip_ref.extract(nc_files[0], temp_file.parent)
                    nc_file = temp_file.parent / nc_files[0]

        # Load and process
        logger.info("Processing NetCDF data...")
        ds = xr.open_dataset(nc_file)

        # Rename valid_time to time if present
        if 'valid_time' in ds.dims:
            ds = ds.rename({'valid_time': 'time'})

        # Convert geopotential to geopotential height (m)
        if 'z' in ds:
            ds['z'] = ds['z'] / 9.80665
            ds['z'].attrs['units'] = 'm'
            ds['z'].attrs['long_name'] = 'Geopotential Height'

        # Save to cache
        logger.info(f"Saving to cache: {cache_file}")
        ds.to_netcdf(cache_file)

        # Cleanup
        if temp_file.exists():
            temp_file.unlink()
        if nc_file != temp_file and nc_file.exists():
            nc_file.unlink()

        logger.info(f"✓ Download complete: {cache_file}")
        logger.info(f"  File size: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")

        return cache_file

    except Exception as e:
        logger.error(f"Download failed: {e}")
        if temp_file.exists():
            temp_file.unlink()
        raise


def _estimate_size(start_year, end_year, spatial_resolution, temporal_frequency, hemisphere):
    """Estimate download size."""
    years = end_year - start_year + 1

    if temporal_frequency == 'hourly':
        timesteps = years * 365.25 * 24
    elif temporal_frequency == '12hourly':
        timesteps = years * 365.25 * 2
    else:  # daily
        timesteps = years * 365.25

    if hemisphere != 'global':
        spatial_factor = 0.5
    else:
        spatial_factor = 1.0

    if '0.25' in spatial_resolution:
        grid_points = 1440 * 721 * spatial_factor
    else:  # 1°
        grid_points = 360 * 181 * spatial_factor

    # Rough estimate: 4 bytes per float32 value
    size_bytes = timesteps * grid_points * 4
    size_mb = size_bytes / 1024 / 1024

    if size_mb < 1024:
        return f"{size_mb:.0f} MB"
    else:
        return f"{size_mb/1024:.1f} GB"


def download_conus_daily_precip_chunk(
    start_year,
    end_year,
    spatial_resolution='0.25/0.25',  # ERA5 native resolution
    time_zone='utc+00:00',
    frequency='1_hourly',  # sub-daily sampling for daily aggregation
    months=None
):
    """
    Download a multi-year chunk of ERA5 daily total precipitation for CONUS.

    Uses the CDS dataset "derived-era5-single-levels-daily-statistics" and
    the variable "total_precipitation" with daily_sum aggregation.

    Args:
        start_year: Start year (e.g., 1940)
        end_year: End year (e.g., 1949)
        spatial_resolution: Grid resolution ('0.25/0.25' = 0.25°, '1.0/1.0' = 1°)
        time_zone: Daily aggregation time zone (default UTC)
        frequency: Sub-daily sampling of the original data ('1_hourly', '3_hourly', '6_hourly')

    Returns:
        Path to cached NetCDF file
    """
    res_name = spatial_resolution.replace('/', 'x').replace('.', 'p')
    month_tag = ""
    if months:
        try:
            m_min = min(months)
            m_max = max(months)
            month_tag = f"_M{m_min:02d}-{m_max:02d}" if m_min != m_max else f"_M{m_min:02d}"
        except Exception:
            month_tag = ""
    cache_file = CACHE_DIR_DAILY_PRECIP / f"era5_tp_daily_CONUS_{start_year}-{end_year}{month_tag}_{res_name}.nc"

    if cache_file.exists():
        logger.info(f"Using cached file: {cache_file}")
        return cache_file

    logger.info(f"Downloading ERA5 daily precipitation (CONUS): {start_year}-{end_year}")
    logger.info(f"  Area: CONUS {CONUS_AREA}")
    logger.info(f"  Resolution: {spatial_resolution}")
    logger.info(f"  Aggregation: daily_sum ({frequency}, {time_zone})")

    years = [str(y) for y in range(start_year, end_year + 1)]
    month_list = months if months else list(range(1, 13))

    c = cdsapi.Client()
    temp_file = cache_file.with_suffix('.nc.tmp')

    try:
        c.retrieve(
            'derived-era5-single-levels-daily-statistics',
            {
                'product_type': 'reanalysis',
                'variable': ['total_precipitation'],
                'year': years,
                'month': [f"{m:02d}" for m in month_list],
                'day': [f"{d:02d}" for d in range(1, 32)],
                'daily_statistic': 'daily_sum',
                'time_zone': time_zone,
                'frequency': frequency,
                'area': CONUS_AREA,
                'grid': [spatial_resolution.split('/')[0], spatial_resolution.split('/')[1]],
                'format': 'netcdf',
            },
            str(temp_file)
        )

        nc_file = temp_file
        if zipfile.is_zipfile(temp_file):
            logger.info("Extracting NetCDF from ZIP...")
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                if nc_files:
                    zip_ref.extract(nc_files[0], temp_file.parent)
                    nc_file = temp_file.parent / nc_files[0]

        logger.info("Processing NetCDF data...")
        ds = xr.open_dataset(nc_file)

        # Rename valid_time to time if present
        if 'valid_time' in ds.dims:
            ds = ds.rename({'valid_time': 'time'})

        # Convert total precipitation from meters to millimeters if present
        if 'tp' in ds:
            ds['tp'] = ds['tp'] * 1000.0
            ds['tp'].attrs['units'] = 'mm'
            ds['tp'].attrs['long_name'] = 'Total Precipitation (daily sum)'

        logger.info(f"Saving to cache: {cache_file}")
        ds.to_netcdf(cache_file)

        if temp_file.exists():
            temp_file.unlink()
        if nc_file != temp_file and nc_file.exists():
            nc_file.unlink()

        logger.info(f"✓ Download complete: {cache_file}")
        logger.info(f"  File size: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")

        return cache_file

    except Exception as e:
        logger.error(f"Download failed: {e}")
        if temp_file.exists():
            temp_file.unlink()
        raise


def download_conus_daily_temp_chunk(
    start_year,
    end_year,
    spatial_resolution='0.25/0.25',
    time_zone='utc+00:00',
    frequency='1_hourly',
    months=None
):
    """
    Download ERA5 daily average 2m temperature for CONUS.

    Uses the CDS dataset "derived-era5-single-levels-daily-statistics" and
    the variable "2m_temperature" with daily_mean aggregation.

    Args:
        start_year: Start year (e.g., 1990)
        end_year: End year (e.g., 2020)
        spatial_resolution: Grid resolution ('0.25/0.25' = 0.25°, '1.0/1.0' = 1°)
        time_zone: Daily aggregation time zone (default UTC)
        frequency: Sub-daily sampling of the original data ('1_hourly', '3_hourly', '6_hourly')
        months: Optional list of months to download (e.g., [1, 2, 3] for Jan-Mar)

    Returns:
        Path to cached NetCDF file
    """
    res_name = spatial_resolution.replace('/', 'x').replace('.', 'p')
    month_tag = ""
    if months:
        try:
            m_min = min(months)
            m_max = max(months)
            month_tag = f"_M{m_min:02d}-{m_max:02d}" if m_min != m_max else f"_M{m_min:02d}"
        except Exception:
            month_tag = ""
    cache_file = CACHE_DIR_DAILY_TEMP / f"era5_t2m_daily_CONUS_{start_year}-{end_year}{month_tag}_{res_name}.nc"

    if cache_file.exists():
        logger.info(f"Using cached file: {cache_file}")
        return cache_file

    logger.info(f"Downloading ERA5 daily 2m temperature (CONUS): {start_year}-{end_year}")
    logger.info(f"  Area: CONUS {CONUS_AREA}")
    logger.info(f"  Resolution: {spatial_resolution}")
    logger.info(f"  Aggregation: daily_mean ({frequency}, {time_zone})")

    years = [str(y) for y in range(start_year, end_year + 1)]
    month_list = months if months else list(range(1, 13))

    c = cdsapi.Client()
    temp_file = cache_file.with_suffix('.nc.tmp')

    try:
        c.retrieve(
            'derived-era5-single-levels-daily-statistics',
            {
                'product_type': 'reanalysis',
                'variable': ['2m_temperature'],
                'year': years,
                'month': [f"{m:02d}" for m in month_list],
                'day': [f"{d:02d}" for d in range(1, 32)],
                'daily_statistic': 'daily_mean',
                'time_zone': time_zone,
                'frequency': frequency,
                'area': CONUS_AREA,
                'grid': [spatial_resolution.split('/')[0], spatial_resolution.split('/')[1]],
                'format': 'netcdf',
            },
            str(temp_file)
        )

        nc_file = temp_file
        if zipfile.is_zipfile(temp_file):
            logger.info("Extracting NetCDF from ZIP...")
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                if nc_files:
                    zip_ref.extract(nc_files[0], temp_file.parent)
                    nc_file = temp_file.parent / nc_files[0]

        logger.info("Processing NetCDF data...")
        ds = xr.open_dataset(nc_file)

        # Rename valid_time to time if present
        if 'valid_time' in ds.dims:
            ds = ds.rename({'valid_time': 'time'})

        # Convert temperature from Kelvin to Celsius for easier interpretation
        if 't2m' in ds:
            ds['t2m'] = ds['t2m'] - 273.15
            ds['t2m'].attrs['units'] = 'degC'
            ds['t2m'].attrs['long_name'] = '2m Temperature (daily mean)'

        logger.info(f"Saving to cache: {cache_file}")
        ds.to_netcdf(cache_file)

        if temp_file.exists():
            temp_file.unlink()
        if nc_file != temp_file and nc_file.exists():
            nc_file.unlink()

        logger.info(f"✓ Download complete: {cache_file}")
        logger.info(f"  File size: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")

        return cache_file

    except Exception as e:
        logger.error(f"Download failed: {e}")
        if temp_file.exists():
            temp_file.unlink()
        raise


def download_full_climatology(
    start_year=1940,
    end_year=2026,
    chunk_years=10,
    **kwargs
):
    """
    Download the full ERA5 climatology in chunks.

    Args:
        start_year: Start year (default: 1940)
        end_year: End year (default: 2026)
        chunk_years: Years per chunk (default: 10)
        **kwargs: Passed to download_global_500mb_chunk

    Returns:
        List of cache file paths
    """
    cache_files = []

    for year in range(start_year, end_year + 1, chunk_years):
        chunk_end = min(year + chunk_years - 1, end_year)

        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading chunk: {year}-{chunk_end}")
        logger.info(f"{'='*60}")

        try:
            cache_file = download_global_500mb_chunk(year, chunk_end, **kwargs)
            cache_files.append(cache_file)

            logger.info(f"✓ Chunk {year}-{chunk_end} complete")
            logger.info(f"  Progress: {len(cache_files)}/{(end_year-start_year+1)//chunk_years + 1} chunks")

        except Exception as e:
            logger.error(f"✗ Chunk {year}-{chunk_end} failed: {e}")
            logger.info("Continuing with next chunk...")
            continue

    logger.info(f"\n{'='*60}")
    logger.info(f"Download complete!")
    logger.info(f"  Successful chunks: {len(cache_files)}")
    logger.info(f"  Cache directory: {CACHE_DIR}")
    logger.info(f"{'='*60}")

    return cache_files


def combine_chunks(cache_files):
    """
    Combine multiple NetCDF chunks into a single dataset.

    Args:
        cache_files: List of NetCDF file paths

    Returns:
        xarray.Dataset with combined data
    """
    logger.info("Combining chunks...")
    datasets = [xr.open_dataset(f) for f in cache_files]
    combined = xr.concat(datasets, dim='time')
    combined = combined.sortby('time')
    logger.info(f"✓ Combined dataset: {len(combined.time)} timesteps")
    return combined


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download ERA5 datasets or compute climatology')
    parser.add_argument('--dataset', type=str, default='z500',
                        choices=['z500', 'precip', 'temp', 'climatology'],
                        help='Dataset to download or "climatology" to compute (default: z500)')
    parser.add_argument('--start-year', type=int, default=2021,
                        help='Start year (default: 2021)')
    parser.add_argument('--end-year', type=int, default=2026,
                        help='End year (default: 2026)')
    parser.add_argument('--chunk-years', type=int, default=0,
                        help='Split downloads into N-year chunks (default: 0 = no chunking)')
    parser.add_argument('--chunk-months', type=int, default=0,
                        help='Split precip downloads into N-month chunks (default: 0 = no chunking)')
    parser.add_argument('--resolution', type=str, default='0.25/0.25',
                        help='Spatial resolution (default: 0.25/0.25; use 1.0/1.0 for z500 if desired)')

    # z500 options
    parser.add_argument('--frequency', type=str, default='daily',
                        choices=['hourly', '12hourly', 'daily'],
                        help='z500 temporal frequency (default: daily)')
    parser.add_argument('--hemisphere', type=str, default='northern',
                        choices=['northern', 'southern', 'global'],
                        help='z500 hemisphere (default: northern)')

    # daily precip options
    parser.add_argument('--precip-frequency', type=str, default='1_hourly',
                        choices=['daily', '1_hourly', '3_hourly', '6_hourly'],
                        help='Sub-daily sampling for precip aggregation (default: 1_hourly; "daily" maps to 1_hourly)')
    parser.add_argument('--precip-time-zone', type=str, default='utc+00:00',
                        help='Aggregation time zone for daily precip (default: utc+00:00)')

    args = parser.parse_args()

    try:
        if args.dataset == 'precip':
            print(f"Downloading ERA5 Daily Precip (CONUS): {args.start_year}-{args.end_year}")
            print(f"  Resolution: {args.resolution}")
            print(f"  Frequency: {args.precip_frequency}")
            print(f"  Time Zone: {args.precip_time_zone}")
            if args.chunk_years and args.chunk_years > 0:
                print(f"  Chunking: {args.chunk_years} years per request")
            if args.chunk_months and args.chunk_months > 0:
                print(f"  Chunking: {args.chunk_months} months per request")
            print("-" * 60)

            precip_freq = '1_hourly' if args.precip_frequency == 'daily' else args.precip_frequency
            if args.chunk_months and args.chunk_months > 0:
                cache_files = []
                for year in range(args.start_year, args.end_year + 1):
                    month = 1
                    while month <= 12:
                        chunk_end = min(month + args.chunk_months - 1, 12)
                        cache_file = download_conus_daily_precip_chunk(
                            start_year=year,
                            end_year=year,
                            spatial_resolution=args.resolution,
                            time_zone=args.precip_time_zone,
                            frequency=precip_freq,
                            months=list(range(month, chunk_end + 1))
                        )
                        cache_files.append(cache_file)
                        month = chunk_end + 1

                print(f"\n✓ SUCCESS! Files saved to:")
                for f in cache_files:
                    print(f"  {f}")
            elif args.chunk_years and args.chunk_years > 0:
                cache_files = []
                y = args.start_year
                while y <= args.end_year:
                    chunk_end = min(y + args.chunk_years - 1, args.end_year)
                    cache_file = download_conus_daily_precip_chunk(
                        start_year=y,
                        end_year=chunk_end,
                        spatial_resolution=args.resolution,
                        time_zone=args.precip_time_zone,
                        frequency=precip_freq
                    )
                    cache_files.append(cache_file)
                    y = chunk_end + 1

                print(f"\n✓ SUCCESS! Files saved to:")
                for f in cache_files:
                    print(f"  {f}")
            else:
                cache_file = download_conus_daily_precip_chunk(
                    start_year=args.start_year,
                    end_year=args.end_year,
                    spatial_resolution=args.resolution,
                    time_zone=args.precip_time_zone,
                    frequency=precip_freq
                )

                print(f"\n✓ SUCCESS! File saved to:")
                print(f"  {cache_file}")
                print(f"  Size: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")
        elif args.dataset == 'temp':
            print(f"Downloading ERA5 Daily 2m Temperature (CONUS): {args.start_year}-{args.end_year}")
            print(f"  Resolution: {args.resolution}")
            print(f"  Frequency: {args.precip_frequency}")
            print(f"  Time Zone: {args.precip_time_zone}")
            if args.chunk_years and args.chunk_years > 0:
                print(f"  Chunking: {args.chunk_years} years per request")
            if args.chunk_months and args.chunk_months > 0:
                print(f"  Chunking: {args.chunk_months} months per request")
            print("-" * 60)

            temp_freq = '1_hourly' if args.precip_frequency == 'daily' else args.precip_frequency
            if args.chunk_months and args.chunk_months > 0:
                cache_files = []
                for year in range(args.start_year, args.end_year + 1):
                    month = 1
                    while month <= 12:
                        chunk_end = min(month + args.chunk_months - 1, 12)
                        cache_file = download_conus_daily_temp_chunk(
                            start_year=year,
                            end_year=year,
                            spatial_resolution=args.resolution,
                            time_zone=args.precip_time_zone,
                            frequency=temp_freq,
                            months=list(range(month, chunk_end + 1))
                        )
                        cache_files.append(cache_file)
                        month = chunk_end + 1

                print(f"\n✓ SUCCESS! Files saved to:")
                for f in cache_files:
                    print(f"  {f}")
            elif args.chunk_years and args.chunk_years > 0:
                cache_files = []
                y = args.start_year
                while y <= args.end_year:
                    chunk_end = min(y + args.chunk_years - 1, args.end_year)
                    cache_file = download_conus_daily_temp_chunk(
                        start_year=y,
                        end_year=chunk_end,
                        spatial_resolution=args.resolution,
                        time_zone=args.precip_time_zone,
                        frequency=temp_freq
                    )
                    cache_files.append(cache_file)
                    y = chunk_end + 1

                print(f"\n✓ SUCCESS! Files saved to:")
                for f in cache_files:
                    print(f"  {f}")
            else:
                cache_file = download_conus_daily_temp_chunk(
                    start_year=args.start_year,
                    end_year=args.end_year,
                    spatial_resolution=args.resolution,
                    time_zone=args.precip_time_zone,
                    frequency=temp_freq
                )

                print(f"\n✓ SUCCESS! File saved to:")
                print(f"  {cache_file}")
                print(f"  Size: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")
        elif args.dataset == 'climatology':
            print("Computing CONUS Climatology (14-day windows)")
            print(f"  Climatology period: {args.start_year}-{args.end_year}")
            print(f"  Variables: precipitation, temperature")
            print("-" * 60)

            output_files = compute_conus_climatology(
                variables=['precip', 'temp'],
                clim_start_year=args.start_year,
                clim_end_year=args.end_year
            )

            print("\n" + "=" * 60)
            print("CLIMATOLOGY FILES CREATED:")
            for var, path in output_files.items():
                file_size = path.stat().st_size / 1024 / 1024
                print(f"  {var}: {path} ({file_size:.1f} MB)")
            print("=" * 60)
        else:
            print(f"Downloading ERA5 500mb Heights: {args.start_year}-{args.end_year}")
            print(f"  Resolution: {args.resolution}")
            print(f"  Frequency: {args.frequency}")
            print(f"  Hemisphere: {args.hemisphere}")
            if args.chunk_years and args.chunk_years > 0:
                print(f"  Chunking: {args.chunk_years} years per request")
            print("-" * 60)

            if args.chunk_years and args.chunk_years > 0:
                cache_files = []
                y = args.start_year
                while y <= args.end_year:
                    chunk_end = min(y + args.chunk_years - 1, args.end_year)
                    cache_file = download_global_500mb_chunk(
                        start_year=y,
                        end_year=chunk_end,
                        spatial_resolution=args.resolution,
                        temporal_frequency=args.frequency,
                        hemisphere=args.hemisphere
                    )
                    cache_files.append(cache_file)
                    y = chunk_end + 1

                print(f"\n✓ SUCCESS! Files saved to:")
                for f in cache_files:
                    print(f"  {f}")
            else:
                cache_file = download_global_500mb_chunk(
                    start_year=args.start_year,
                    end_year=args.end_year,
                    spatial_resolution=args.resolution,
                    temporal_frequency=args.frequency,
                    hemisphere=args.hemisphere
                )

                print(f"\n✓ SUCCESS! File saved to:")
                print(f"  {cache_file}")
                print(f"  Size: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")

            # Load and show info for z500
            ds = xr.open_dataset(cache_file)
            print(f"\nDataset info:")
            print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
            print(f"  Timesteps: {len(ds.time)}")
            print(f"  Lat range: {float(ds.latitude.min()):.1f}° to {float(ds.latitude.max()):.1f}°")
            print(f"  Lon range: {float(ds.longitude.min()):.1f}° to {float(ds.longitude.max()):.1f}°")
            print(f"  Grid size: {len(ds.latitude)} × {len(ds.longitude)}")

            if 'z' in ds:
                z_data = ds['z'].isel(pressure_level=0)  # Select 500mb level
                z_mean = float(z_data.mean())
                print(f"\n500mb Height Statistics:")
                print(f"  Mean: {z_mean:.1f} m ({z_mean/9.80665:.0f} dam)")

            print("\n" + "=" * 60)
            print("To backfill earlier years, run:")
            print("  python3 era5_bulk_download.py --dataset z500 --start-year 2016 --end-year 2020")
            print("  python3 era5_bulk_download.py --dataset z500 --start-year 2011 --end-year 2015")
            print("  python3 era5_bulk_download.py --dataset z500 --start-year 2006 --end-year 2010")
            print("  ... and so on back to 1940")
            print("=" * 60)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("\nMake sure you have:")
        print("  1. CDS API credentials configured (~/.cdsapirc)")
        print("  2. Accepted the ERA5 license terms on the CDS website")
        import traceback
        traceback.print_exc()


def compute_conus_climatology(
    variables=['precip', 'temp'],
    clim_start_year=1990,
    clim_end_year=2020,
    window_days=31,
    output_dir=None
):
    """
    Pre-compute 14-day climatology for CONUS grid.

    For each grid point and day-of-year:
    - Collect all 14-day windows starting on that DOY across climatology years
    - Use ±15 day window (31-day total) to smooth seasonal transitions
    - Compute mean as climatological normal

    Args:
        variables: List of variables to compute ['precip', 'temp']
        clim_start_year: Start of climatology period (default 1990)
        clim_end_year: End of climatology period (default 2020)
        window_days: Days around DOY for smoothing (default 31)
        output_dir: Output directory (default: CACHE_DIR_CLIMATOLOGY)

    Returns:
        Dictionary with paths to output NetCDF files
    """
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    if output_dir is None:
        output_dir = CACHE_DIR_CLIMATOLOGY

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("COMPUTING CONUS CLIMATOLOGY (14-DAY WINDOWS)")
    logger.info("=" * 80)
    logger.info(f"Variables: {variables}")
    logger.info(f"Climatology period: {clim_start_year}-{clim_end_year}")
    logger.info(f"Smoothing window: ±{window_days//2} days")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    output_files = {}

    # Process each variable
    for var in variables:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing variable: {var.upper()}")
        logger.info(f"{'='*80}")

        # Determine file pattern and variable name
        if var == 'precip':
            file_pattern = CACHE_DIR_DAILY_PRECIP / "era5_tp_daily_CONUS_*_M*.nc"
            var_name = 'tp'
            output_file = output_dir / 'conus_precip_climatology_14day.nc'
            long_name = '14-day Precipitation Climatology'
            units = 'mm'
            aggregation = 'sum'
        elif var == 'temp':
            file_pattern = CACHE_DIR_DAILY_TEMP / "era5_t2m_daily_CONUS_*_M*.nc"
            var_name = 't2m'
            output_file = output_dir / 'conus_temp_climatology_14day.nc'
            long_name = '14-day Temperature Climatology'
            units = 'degC'
            aggregation = 'mean'
        else:
            logger.warning(f"Unknown variable: {var}, skipping")
            continue

        # Find all matching files
        import glob
        all_files = sorted(glob.glob(str(file_pattern)))

        if not all_files:
            logger.error(f"No files found matching pattern: {file_pattern}")
            logger.error(f"Please download data first using:")
            logger.error(f"  python3 era5_bulk_download.py --dataset {var} --start-year {clim_start_year} --end-year {clim_end_year} --chunk-months 1")
            continue

        logger.info(f"Found {len(all_files)} data files")

        # Load all data using open_mfdataset
        logger.info("Loading dataset (this may take a minute)...")
        try:
            ds_all = xr.open_mfdataset(
                all_files,
                combine='by_coords',
                parallel=True,
                engine='netcdf4'
            )

            # Filter to climatology period
            ds_all = ds_all.sel(time=slice(f"{clim_start_year}-01-01", f"{clim_end_year}-12-31"))

            logger.info(f"  Time range: {ds_all.time.values[0]} to {ds_all.time.values[-1]}")
            logger.info(f"  Grid shape: {len(ds_all.latitude)} lat × {len(ds_all.longitude)} lon")
            logger.info(f"  Total timesteps: {len(ds_all.time)}")

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            continue

        # Extract coordinate arrays
        lats = ds_all.latitude.values
        lons = ds_all.longitude.values

        # Initialize output array: (366 DOY, lat, lon)
        clim_data = np.full((366, len(lats), len(lons)), np.nan, dtype=np.float32)

        logger.info("\nComputing climatology for each day-of-year...")
        logger.info("(This will take 10-20 minutes...)")

        # Convert times to pandas DatetimeIndex for easier DOY handling
        times = pd.to_datetime(ds_all.time.values)

        # Process each day-of-year
        for doy in tqdm(range(1, 367), desc=f"Computing {var} climatology"):
            # Find all dates in the climatology period that match this DOY
            # Handle leap years: if DOY > 59 (after Feb 28), map Feb 29 to Feb 28
            matching_dates = []

            for year in range(clim_start_year, clim_end_year + 1):
                try:
                    # Try to create the date
                    if doy <= 59:  # Jan 1 - Feb 28
                        date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
                    elif doy == 60 and pd.Timestamp(year=year, month=1, day=1).is_leap_year:
                        # Feb 29 in leap year
                        date = pd.Timestamp(year=year, month=2, day=29)
                    else:
                        # After Feb 28/29
                        if pd.Timestamp(year=year, month=1, day=1).is_leap_year:
                            date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
                        else:
                            # Non-leap year: skip DOY 60 or adjust
                            if doy == 60:
                                continue  # Skip Feb 29 for non-leap years
                            date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 2)

                    matching_dates.append(date)
                except:
                    continue

            if not matching_dates:
                logger.warning(f"No matching dates for DOY {doy}")
                continue

            # For each matching date, compute 14-day window
            windows_14day = []

            for start_date in matching_dates:
                end_date = start_date + pd.Timedelta(days=13)  # 14 days total (inclusive of start)

                # Check if data exists for this window
                window_data = ds_all[var_name].sel(time=slice(start_date, end_date))

                if len(window_data.time) >= 10:  # Need at least 10 days of data
                    # Aggregate the 14-day window
                    if aggregation == 'sum':
                        window_value = window_data.sum(dim='time')
                    else:  # mean
                        window_value = window_data.mean(dim='time')

                    windows_14day.append(window_value.values)

            if windows_14day:
                # Average across all years
                clim_data[doy - 1, :, :] = np.nanmean(windows_14day, axis=0)

        # Create output dataset
        logger.info("\nCreating output NetCDF file...")

        ds_out = xr.Dataset(
            {
                f'{var}_14d_clim': (['doy', 'latitude', 'longitude'], clim_data, {
                    'long_name': long_name,
                    'units': units,
                    'description': f'{aggregation.capitalize()} of 14-day windows starting on each DOY, averaged across {clim_start_year}-{clim_end_year}',
                })
            },
            coords={
                'doy': (['doy'], np.arange(1, 367), {
                    'long_name': 'Day of Year',
                    'description': 'Day of year (1-366, includes Feb 29 for leap years)'
                }),
                'latitude': (['latitude'], lats, {
                    'units': 'degrees_north',
                    'long_name': 'Latitude'
                }),
                'longitude': (['longitude'], lons, {
                    'units': 'degrees_east',
                    'long_name': 'Longitude'
                })
            },
            attrs={
                'title': f'CONUS {var.upper()} 14-day Climatology',
                'climatology_period': f'{clim_start_year}-{clim_end_year}',
                'smoothing_window_days': window_days,
                'created': pd.Timestamp.now().isoformat(),
                'description': f'14-day {aggregation} climatology for CONUS, computed from ERA5 reanalysis'
            }
        )

        # Save to NetCDF
        logger.info(f"Saving to: {output_file}")
        ds_out.to_netcdf(output_file, encoding={
            f'{var}_14d_clim': {'dtype': 'float32', 'zlib': True, 'complevel': 5}
        })

        file_size_mb = output_file.stat().st_size / 1024 / 1024
        logger.info(f"✓ Climatology file created: {output_file}")
        logger.info(f"  File size: {file_size_mb:.1f} MB")

        output_files[var] = output_file

        # Cleanup
        ds_all.close()

    logger.info("\n" + "=" * 80)
    logger.info("CLIMATOLOGY COMPUTATION COMPLETE")
    logger.info("=" * 80)
    for var, path in output_files.items():
        logger.info(f"  {var}: {path}")
    logger.info("=" * 80)

    return output_files
