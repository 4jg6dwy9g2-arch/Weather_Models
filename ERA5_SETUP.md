# ERA5 Reanalysis Setup Guide

## What is ERA5?

ERA5 is ECMWF's latest climate reanalysis dataset, providing:
- **Time coverage**: 1940 to near real-time (5-day delay)
- **Temporal resolution**: Hourly data
- **Spatial resolution**: 0.25° x 0.25° (~31 km)
- **Variables**: 100+ atmospheric, land, and ocean variables
- **Data assimilation**: Combines models with observations for "best estimate" of past weather

## Setup Instructions

### 1. Register for CDS API Access

1. Go to: https://cds.climate.copernicus.eu/user/register
2. Register for a free account
3. Verify your email address
4. Log in to the CDS portal

### 2. Get API Credentials

1. Go to: https://cds.climate.copernicus.eu/api-how-to
2. You'll see your UID and API key at the bottom
3. Copy these credentials

### 3. Install Python Package

```bash
pip install cdsapi
```

### 4. Configure API Access

Create a file at `~/.cdsapirc` (in your home directory) with:

```
url: https://cds.climate.copernicus.eu/api
key: {YOUR-API-KEY}
```

**IMPORTANT**: Use the NEW API format (without `/v2` and without `UID:` prefix).

Replace `{YOUR-API-KEY}` with your actual API key from step 2 (just the key part, not the UID).

**macOS/Linux:**
```bash
nano ~/.cdsapirc
```

Then paste these two lines (replacing YOUR_API_KEY with your actual key):
```
url: https://cds.climate.copernicus.eu/api
key: YOUR_API_KEY
```

Save (Ctrl+X, Y, Enter) and set permissions:
```bash
chmod 600 ~/.cdsapirc
```

**Windows:**
Create the file at: `C:\Users\YourUsername\.cdsapirc`

### 5. Test the Installation

```bash
cd /Users/kennypratt/Documents/Weather_Models
python3 era5.py
```

This will fetch the last 30 days of data and cache it locally.

## Usage Examples

### Get Surface Data

```python
from era5 import get_era5_surface, extract_point_timeseries, era5_to_dataframe
from datetime import datetime, timedelta

# Fetch last 7 days
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

ds = get_era5_surface(start_date, end_date)

# Extract data for Fairfax location
point_ds = extract_point_timeseries(ds, 38.8419, -77.3091)

# Convert to DataFrame
df = era5_to_dataframe(point_ds)
print(df.head())
```

### Get Daily Summary

```python
from era5 import get_daily_summary
from datetime import datetime

# Get January 2024 daily data
daily = get_daily_summary('2024-01-01', '2024-01-31')
print(daily)
```

### Get 500mb Heights for Rossby Wave Analysis

```python
from era5 import get_500mb_history

# Get 500mb heights for last year
heights = get_500mb_history('2024-01-01', '2024-12-31')
print(heights)
```

### Get Upper Air Data

```python
from era5 import get_era5_pressure_levels

# Get multiple pressure levels
ds = get_era5_pressure_levels(
    '2024-01-01', '2024-01-31',
    levels=[1000, 850, 500, 250],
    variables=['temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind']
)
```

### Calculate Long-term Climatology

```python
from era5 import get_climatology

# Get 1991-2020 climatology
climo = get_climatology(1991, 2020)
print(climo)
```

## Available Variables

### Surface (Single-Level) Variables

| Variable Name | Description | Units (API) | Units (Converted) |
|--------------|-------------|-------------|-------------------|
| `2m_temperature` | 2m temperature | K | °F |
| `2m_dewpoint_temperature` | 2m dewpoint | K | °F |
| `surface_pressure` | Surface pressure | Pa | mb |
| `mean_sea_level_pressure` | MSLP | Pa | mb |
| `total_precipitation` | Total precip | m | inches |
| `10m_u_component_of_wind` | U wind | m/s | m/s |
| `10m_v_component_of_wind` | V wind | m/s | m/s |
| `surface_solar_radiation_downwards` | Solar radiation | J/m² | J/m² |
| `total_cloud_cover` | Cloud cover | 0-1 | 0-1 |
| `skin_temperature` | Skin temp | K | °F |
| `snow_depth` | Snow depth | m | m |

### Pressure-Level Variables

| Variable Name | Description | Levels (hPa) |
|--------------|-------------|--------------|
| `geopotential` | Geopotential height | 1000-1 |
| `temperature` | Air temperature | 1000-1 |
| `u_component_of_wind` | U wind component | 1000-1 |
| `v_component_of_wind` | V wind component | 1000-1 |
| `relative_humidity` | Relative humidity | 1000-1 |
| `specific_humidity` | Specific humidity | 1000-1 |
| `vertical_velocity` | Omega | 1000-1 |

Full variable list: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels

## Data Caching

- All data is automatically cached in `data/era5/` directory
- Subsequent requests for the same date range use cached data
- Cache files are NetCDF format (.nc)
- Delete cache files to force re-download

## API Limits

- **Free tier limits**:
  - 1 request at a time
  - ~10,000 requests per day
  - Data download limits (varies)
- **Best practices**:
  - Request larger time chunks instead of many small requests
  - Use caching to avoid re-downloading
  - Request only the variables and area you need
  - Consider downloading full years at once for climatology

## Common Issues

### "Connection refused" or "401 Unauthorized"
- Check your `~/.cdsapirc` file exists and has correct credentials
- Verify you've accepted the Terms & Conditions on the CDS website

### "Request too large"
- Reduce the time period or spatial area
- Request fewer variables
- Break into multiple smaller requests

### Slow downloads
- ERA5 API can be slow during peak hours
- First-time requests are slower (data retrieval from tape)
- Use cache to avoid re-downloading

## Integration Ideas

1. **Historical Weather Page**: Show ERA5 data alongside WeatherLink
2. **Gap Filling**: Use ERA5 when WeatherLink is offline
3. **Extended Climatology**: Replace IAD climatology with ERA5
4. **Upper Air Analysis**: Add 500mb height history
5. **Weather Analogs**: Find similar historical patterns
6. **Model Verification**: Use ERA5 as "truth" for upper air

## Resources

- **CDS Portal**: https://cds.climate.copernicus.eu
- **ERA5 Documentation**: https://confluence.ecmwf.int/display/CKB/ERA5
- **Python API Docs**: https://cds.climate.copernicus.eu/api-how-to
- **Variable Catalog**: https://codes.ecmwf.int/grib/param-db/
