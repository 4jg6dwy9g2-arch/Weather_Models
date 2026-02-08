"""
ERA5 Bulk Download Script for Global 500mb Geopotential Heights
Optimized for downloading large historical datasets (1940-present)
"""

import cdsapi
import xarray as xr
from pathlib import Path
from datetime import datetime
import logging
import zipfile

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cache directory on T7 drive
CACHE_DIR = Path("/Volumes/T7/Weather_Models/era5/global_500mb")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
    import argparse

    parser = argparse.ArgumentParser(description='Download ERA5 500mb geopotential heights')
    parser.add_argument('--start-year', type=int, default=2021,
                        help='Start year (default: 2021)')
    parser.add_argument('--end-year', type=int, default=2026,
                        help='End year (default: 2026)')
    parser.add_argument('--resolution', type=str, default='1.0/1.0',
                        help='Spatial resolution (default: 1.0/1.0)')
    parser.add_argument('--frequency', type=str, default='daily',
                        choices=['hourly', '12hourly', 'daily'],
                        help='Temporal frequency (default: daily)')
    parser.add_argument('--hemisphere', type=str, default='northern',
                        choices=['northern', 'southern', 'global'],
                        help='Hemisphere (default: northern)')

    args = parser.parse_args()

    print(f"Downloading ERA5 500mb Heights: {args.start_year}-{args.end_year}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Frequency: {args.frequency}")
    print(f"  Hemisphere: {args.hemisphere}")
    print("-" * 60)

    try:
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

        # Load and show info
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
        print(f"  python3 era5_bulk_download.py --start-year 2016 --end-year 2020")
        print(f"  python3 era5_bulk_download.py --start-year 2011 --end-year 2015")
        print(f"  python3 era5_bulk_download.py --start-year 2006 --end-year 2010")
        print("  ... and so on back to 1940")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("\nMake sure you have:")
        print("  1. CDS API credentials configured (~/.cdsapirc)")
        print("  2. Accepted the ERA5 license terms on the CDS website")
        import traceback
        traceback.print_exc()
