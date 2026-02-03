"""
ECMWF IFS (Integrated Forecasting System) model implementation.

IFS is ECMWF's operational physics-based numerical weather prediction model.
Data is freely available via ECMWF Open Data (no authentication required).

Install: pip install ecmwf-opendata
"""

from datetime import datetime, timedelta, timezone

from pathlib import Path

from typing import Dict, List, Optional, Tuple, Union

import logging
import tempfile
from rate_limiter import RateLimiter # Import the RateLimiter

import numpy as np
import xarray as xr

from base import WeatherModel

# Instantiate a rate limiter for ECMWF Open Data calls (e.g., 1 call every 2 seconds)
ecmwf_rate_limiter = RateLimiter(calls_per_second=1)

CACHE_DIR = Path.home() / ".cache" / "weather_models"

CACHE_DIR.mkdir(parents=True, exist_ok=True)



logger = logging.getLogger(__name__)





class Variable:

    def __init__(self, name, display_name, units, ecmwf_param, category, colormap, contour_levels, fill=True, level=None):

        self.name = name

        self.display_name = display_name

        self.units = units

        self.ecmwf_param = ecmwf_param  # ECMWF parameter name

        self.category = category

        self.colormap = colormap

        self.contour_levels = contour_levels

        self.fill = fill

        self.level = level





class Region:

    def __init__(self, name, bounds):

        self.name = name

        self.bounds = bounds





# IFS variables available from ECMWF Open Data

IFS_VARIABLES: Dict[str, Variable] = {

    "t2m": Variable(

        name="t2m",

        display_name="2m Temperature",

        units="F",

        ecmwf_param="2t",

        category="surface",

        colormap="RdYlBu_r",

        contour_levels=list(range(-40, 120, 5))

    ),

    "mslp": Variable(

        name="mslp",

        display_name="Mean Sea Level Pressure",

        units="mb",

        ecmwf_param="msl",

        category="surface",

        colormap="coolwarm",

        contour_levels=list(range(960, 1060, 4)),

        fill=False

    ),

    "tp": Variable(

        name="tp",

        display_name="Total Precipitation",

        units="in",

        ecmwf_param="tp",

        category="surface",

        colormap="precip",

        contour_levels=[0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

    ),

    "z500": Variable(

        name="z500",

        display_name="500 hPa Geopotential Height",

        units="dm",

        ecmwf_param="gh",

        category="upper_air",

        colormap="viridis",

        contour_levels=list(range(480, 600, 6)),

        level="500"

    ),

}



# IFS forecast hours (6-hourly out to 10 days via open data)

IFS_FORECAST_HOURS = list(range(0, 241, 6))



# IFS runs at 00Z, 06Z, 12Z, 18Z

IFS_INIT_HOURS = [0, 6, 12, 18]





class IFSModel(WeatherModel):

    """ECMWF IFS model data access via ecmwf-opendata."""



    def __init__(self):

        super().__init__("IFS")

        self._client = None

        self._download_dir = CACHE_DIR / "ifs_downloads"

        self._download_dir.mkdir(parents=True, exist_ok=True)



    def _get_client(self):

        """Get or create ECMWF Open Data client for IFS."""

        if self._client is None:

            try:

                                from ecmwf.opendata import Client

                                self._client = Client(source="aws", model="ifs")

            except ImportError:

                raise ImportError(

                    "ecmwf-opendata package required. Install with: pip install ecmwf-opendata"

                )

        return self._client



    @property

    def available_variables(self) -> Dict[str, Variable]:

        """Return available IFS variables."""

        return IFS_VARIABLES



    @property

    def forecast_hours(self) -> List[int]:

        """Return IFS forecast hours."""

        return IFS_FORECAST_HOURS



    @property

    def init_hours(self) -> List[int]:

        """Return IFS initialization hours."""

        return IFS_INIT_HOURS



    def get_latest_init_time(self) -> datetime:

        """

        Get the most recent available IFS initialization time.

        """

        now = datetime.now(timezone.utc)



        # IFS data typically available ~6 hours after init time

        for hours_back in range(6, 48, 1):

            candidate = now - timedelta(hours=hours_back)

            candidate = candidate.replace(minute=0, second=0, microsecond=0)



            if candidate.hour in self.init_hours:

                return candidate



        # Fallback

        yesterday = now - timedelta(days=1)

        return yesterday.replace(hour=0, minute=0, second=0, microsecond=0)

    def get_init_time_for_hour(self, init_hour: int) -> datetime:
        """
        Get the initialization time for a specific hour today (or yesterday if not ready yet).

        Args:
            init_hour: The init hour (0, 6, 12, or 18)

        Returns:
            datetime: The init time for the specified hour
        """
        if init_hour not in self.init_hours:
            raise ValueError(f"Invalid init hour {init_hour}. Must be one of {self.init_hours}")

        now = datetime.now(timezone.utc)
        today = now.replace(hour=init_hour, minute=0, second=0, microsecond=0)

        # IFS data typically available ~6 hours after init time
        hours_since_init = (now - today).total_seconds() / 3600

        if hours_since_init >= 6:
            return today
        else:
            # Use yesterday's run at this hour
            return today - timedelta(days=1)

    def get_available_init_times(self, days_back: int = 3) -> List[datetime]:
        """Get list of available initialization times."""
        init_times = []
        latest = self.get_latest_init_time()

        current = latest
        end_time = latest - timedelta(days=days_back)

        while current >= end_time:
            init_times.append(current)
            current = current - timedelta(hours=6)
            while current.hour not in self.init_hours:
                current = current - timedelta(hours=1)

        return init_times

    @ecmwf_rate_limiter
    def _download_grib(
        self,
        init_time: datetime,
        forecast_hour: int,
        param: str,
        level: Optional[str] = None,
    ) -> Path:
        """Download GRIB file from ECMWF Open Data."""
        client = self._get_client()

        # Build filename
        level_str = f"_{level}" if level else ""
        filename = f"ifs_{init_time.strftime('%Y%m%d%H')}_{forecast_hour:03d}_{param}{level_str}.grib2"
        filepath = self._download_dir / filename

        # Return cached file if exists
        if filepath.exists():
            logger.info(f"Using cached GRIB: {filename}")
            return filepath

        # Convert to naive UTC datetime if timezone-aware
        if init_time.tzinfo is not None:
            init_time_naive = init_time.replace(tzinfo=None)
        else:
            init_time_naive = init_time

        logger.info(f"Downloading IFS {param} for F{forecast_hour:03d} from {init_time_naive.strftime('%Y%m%d %HZ')}")

        try:
            kwargs = {
                "date": init_time_naive.strftime('%Y%m%d'),
                "time": init_time_naive.hour,
                "step": forecast_hour,
                "param": param,
                "type": "fc",
                "target": str(filepath.absolute()),
            }

            if level:
                kwargs["levelist"] = int(level)

            client.retrieve(**kwargs)

            if not filepath.exists():
                raise FileNotFoundError(f"Download completed but file not found: {filepath}")

            logger.info(f"Downloaded {filepath.name} ({filepath.stat().st_size} bytes)")
            return filepath

        except Exception as e:
            logger.error(f"Failed to download IFS data: {e}")
            raise

    @ecmwf_rate_limiter
    def _download_grib_batch(
        self,
        init_time: datetime,
        forecast_hour: int,
        params: List[str],
        level: Optional[Union[str, int]] = None,
    ) -> Path:
        """Download GRIB file with multiple parameters from ECMWF Open Data."""
        client = self._get_client()

        # Generate a filename that includes all parameters for batch download
        param_str = "_".join(sorted(params)) # Sort for consistent filenames
        level_str = f"_{level}" if level else ""
        filename = f"ifs_{init_time.strftime('%Y%m%d%H')}_{forecast_hour:03d}_{param_str}{level_str}.grib2"
        filepath = self._download_dir / filename

        # Return cached file if exists
        if filepath.exists():
            logger.info(f"Using cached batch GRIB: {filename}")
            return filepath

        if init_time.tzinfo is not None:
            init_time_naive = init_time.replace(tzinfo=None)
        else:
            init_time_naive = init_time

        logger.info(f"Downloading IFS batch params ({param_str}) for F{forecast_hour:03d} from {init_time_naive.strftime('%Y%m%d %HZ')}")

        try:
            kwargs = {
                "date": init_time_naive.strftime('%Y%m%d'),
                "time": init_time_naive.hour,
                "step": forecast_hour,
                "param": params, # Pass list of parameters
                "type": "fc",
                "target": str(filepath.absolute()),
            }

            if level:
                kwargs["levelist"] = int(level)

            client.retrieve(**kwargs)

            if not filepath.exists():
                raise FileNotFoundError(f"Batch download completed but file not found: {filepath}")

            logger.info(f"Downloaded batch {filepath.name} ({filepath.stat().st_size} bytes)")
            return filepath

        except Exception as e:
            logger.error(f"Failed to download IFS batch data: {e}")
            raise

    def fetch_data(
        self,
        variable: Variable,
        init_time: datetime,
        forecast_hour: int,
        region: Optional[Region] = None,
    ) -> xr.DataArray:
        """Fetch IFS data for a specific variable."""
        try:
            grib_path = self._download_grib(
                init_time,
                forecast_hour,
                variable.ecmwf_param,
                variable.level,
            )

            ds = xr.open_dataset(
                str(grib_path),
                engine="cfgrib",
                backend_kwargs={"indexpath": ""},
            )

            data_vars = [v for v in ds.data_vars]
            if not data_vars:
                raise ValueError(f"No data found in GRIB file")

            da = ds[data_vars[0]]
            logger.info(f"Loaded {data_vars[0]} with shape {da.shape}")

            da = self._convert_units(da, variable)

            if region is not None:
                da = self._subset_region(da, region)

            da.attrs['variable_name'] = variable.name
            da.attrs['display_name'] = variable.display_name
            da.attrs['init_time'] = init_time.isoformat()
            da.attrs['forecast_hour'] = forecast_hour
            da.attrs['valid_time'] = self.valid_time(init_time, forecast_hour).isoformat()
            da.attrs['model'] = 'ECMWF IFS'

            return da

        except Exception as e:
            logger.error(f"Error fetching IFS {variable.name}: {e}")
            raise

    def _convert_units(self, da: xr.DataArray, variable: Variable) -> xr.DataArray:
        """Apply unit conversions."""
        units = da.attrs.get('units', da.attrs.get('GRIB_units', ''))

        # Temperature: K to F or C
        if variable.units == "F" and units == "K":
            da = (da - 273.15) * 9/5 + 32
            da.attrs['units'] = 'F'
        elif variable.units == "C" and units == "K":
            da = da - 273.15
            da.attrs['units'] = 'C'

        # Pressure: Pa to mb
        elif variable.units == "mb" and units == "Pa":
            da = da / 100
            da.attrs['units'] = 'mb'

        # Precipitation: m to inches, or kg/m² (mm) to inches
        elif variable.units == "in":
            if units == "m":
                da = da * 39.3701
                da.attrs['units'] = 'in'
            elif units in ["kg m**-2", "kg/m^2", "kg/m²", "mm"]:
                da = da / 25.4
                da.attrs['units'] = 'in'

        # Height conversions (m² s⁻² to dm, or m to dm for geopotential height)
        elif variable.units == "dm":
            if units in ["m**2 s**-2", "m^2 s^-2", "m**2/s**2"]:
                # Geopotential to geopotential height: divide by g (9.80665 m/s²), then convert m to dm
                da = da / 9.80665 / 10
                da.attrs['units'] = 'dm'
            elif units == "m":
                da = da / 10
                da.attrs['units'] = 'dm'
            elif units == "gpm":
                da = da / 10
                da.attrs['units'] = 'dm'

        return da

    def _subset_region(self, da: xr.DataArray, region: Region) -> xr.DataArray:
        """Subset data to a geographic region."""
        west, east, south, north = region.bounds

        is_global_lon = (west <= -180 and east >= 180) or (east - west >= 360)
        is_global_lat = (south <= -90 and north >= 90)

        if is_global_lon and is_global_lat:
            return da

        lat_name = 'latitude' if 'latitude' in da.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in da.coords else 'lon'

        if not is_global_lon:
            lons = da[lon_name].values

            if np.any(lons < 0):
                da = da.sel({lon_name: slice(west, east)})
            else:
                west_360 = west % 360 if west < 0 else west
                east_360 = east % 360 if east < 0 else east
                if west_360 > east_360:
                    da_west = da.sel({lon_name: slice(west_360, 360)})
                    da_east = da.sel({lon_name: slice(0, east_360)})
                    da = xr.concat([da_west, da_east], dim=lon_name)
                else:
                    da = da.sel({lon_name: slice(west_360, east_360)})

        if not is_global_lat:
            lat_vals = da[lat_name].values
            if lat_vals[0] > lat_vals[-1]:
                da = da.sel({lat_name: slice(north, south)})
            else:
                da = da.sel({lat_name: slice(south, north)})

        return da
