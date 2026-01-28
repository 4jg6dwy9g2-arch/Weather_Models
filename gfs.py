"""
GFS (Global Forecast System) model implementation.

Uses Herbie for data access from NOMADS/AWS with automatic failover.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from base import WeatherModel

class Variable:
    def __init__(self, name, display_name, units, herbie_search, category, colormap, contour_levels, fill=True, level=None):
        self.name = name
        self.display_name = display_name
        self.units = units
        self.herbie_search = herbie_search
        self.category = category
        self.colormap = colormap
        self.contour_levels = contour_levels
        self.fill = fill
        self.level = level

class Region:
    def __init__(self, name, bounds):
        self.name = name
        self.bounds = bounds

ALL_VARIABLES = {}
GFS_FORECAST_HOURS = list(range(0, 385, 3))
GFS_INIT_HOURS = [0, 6, 12, 18]

logger = logging.getLogger(__name__)


class GFSModel(WeatherModel):
    """GFS model data access via Herbie."""

    def __init__(self):
        super().__init__("GFS")
        self._herbie_available = None

    def _check_herbie(self) -> bool:
        """Check if Herbie is available."""
        if self._herbie_available is None:
            try:
                from herbie import Herbie
                self._herbie_available = True
            except ImportError:
                logger.warning("Herbie not installed. Install with: conda install -c conda-forge herbie-data")
                self._herbie_available = False
        return self._herbie_available

    @property
    def available_variables(self) -> Dict[str, Variable]:
        """Return all available GFS variables."""
        return ALL_VARIABLES

    @property
    def forecast_hours(self) -> List[int]:
        """Return GFS forecast hours (0-384)."""
        return GFS_FORECAST_HOURS

    @property
    def init_hours(self) -> List[int]:
        """Return GFS initialization hours."""
        return GFS_INIT_HOURS

    def get_latest_init_time(self) -> datetime:
        """
        Get the most recent available GFS initialization time.

        GFS runs take ~3.5 hours to complete, so we look back from current time.
        """
        now = datetime.utcnow()

        # Find the most recent init hour that's at least 4 hours old
        for hours_back in range(4, 48, 1):
            candidate = now - timedelta(hours=hours_back)
            candidate = candidate.replace(minute=0, second=0, microsecond=0)

            if candidate.hour in self.init_hours:
                # Round down to the init hour
                return candidate.replace(hour=candidate.hour)

        # Fallback to yesterday's 00Z
        yesterday = now - timedelta(days=1)
        return yesterday.replace(hour=0, minute=0, second=0, microsecond=0)

    def get_available_init_times(self, days_back: int = 3) -> List[datetime]:
        """Get list of available initialization times."""
        init_times = []
        latest = self.get_latest_init_time()

        # Generate init times going back
        current = latest
        end_time = latest - timedelta(days=days_back)

        while current >= end_time:
            init_times.append(current)
            # Go back to previous init time
            current = current - timedelta(hours=6)
            # Snap to valid init hour
            while current.hour not in self.init_hours:
                current = current - timedelta(hours=1)

        return init_times

    def _get_herbie(self, init_time: datetime, forecast_hour: int, save_dir: Path):
        """Create a Herbie object for the given init time and forecast hour."""
        if not self._check_herbie():
            raise ImportError("Herbie is required for GFS data access")

        from herbie import Herbie
        from pathlib import Path
        
        return Herbie(
            init_time,
            model="gfs",
            product="pgrb2.0p25",  # 0.25 degree resolution
            fxx=forecast_hour,
            save_dir=save_dir,
        )

    def fetch_data(
        self,
        variable: Variable,
        init_time: datetime,
        forecast_hour: int,
        region: Optional[Region] = None,
    ) -> xr.DataArray:
        """
        Fetch GFS data for a specific variable.

        Args:
            variable: Variable definition
            init_time: Model initialization time
            forecast_hour: Forecast hour
            region: Optional region for subsetting

        Returns:
            xarray DataArray with the requested data
        """
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_save_dir = Path(tmpdir)
            H = self._get_herbie(init_time, forecast_hour, tmp_save_dir)

            logger.info(f"Fetching {variable.display_name} for {init_time} F{forecast_hour:03d}")

            try:
                # Download and open the GRIB2 data
                ds = H.xarray(variable.herbie_search, remove_grib=True) # remove_grib=True since we are using a temporary directory

                # Get the data variable (first non-coordinate variable)
                data_vars = [v for v in ds.data_vars if v not in ['latitude', 'longitude', 'time', 'step', 'valid_time']]
                if not data_vars:
                    raise ValueError(f"No data found for variable {variable.name}")

                da = ds[data_vars[0]]

                # Apply unit conversions
                da = self._convert_units(da, variable)

                # Subset to region if specified
                if region is not None:
                    da = self._subset_region(da, region)

                # Add metadata
                da.attrs['variable_name'] = variable.name
                da.attrs['display_name'] = variable.display_name
                da.attrs['init_time'] = init_time.isoformat()
                da.attrs['forecast_hour'] = forecast_hour
                da.attrs['valid_time'] = self.valid_time(init_time, forecast_hour).isoformat()

                return da

            except Exception as e:
                logger.error(f"Error fetching {variable.name}: {e}")
                raise

    def fetch_wind_components(
        self,
        level: str,
        init_time: datetime,
        forecast_hour: int,
        region: Optional[Region] = None,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Fetch U and V wind components for a given level.

        Args:
            level: Pressure level ("850", "250", etc.) or "10m" for surface
            init_time: Model initialization time
            forecast_hour: Forecast hour
            region: Optional region for subsetting

        Returns:
            Tuple of (u_component, v_component) DataArrays
        """
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_save_dir = Path(tmpdir)
            H = self._get_herbie(init_time, forecast_hour, tmp_save_dir)

            if level == "10m":
                u_search = ":UGRD:10 m above ground:"
                v_search = ":VGRD:10 m above ground:"
            else:
                u_search = f":UGRD:{level} mb:"
                v_search = f":VGRD:{level} mb:"

            logger.info(f"Fetching wind components at {level} for {init_time} F{forecast_hour:03d}")

            try:
                # Fetch U component
                ds_u = H.xarray(u_search, remove_grib=True)
                u_vars = [v for v in ds_u.data_vars if 'u' in v.lower() or 'ugrd' in v.lower()]
                if not u_vars:
                    u_vars = [v for v in ds_u.data_vars if v not in ['latitude', 'longitude', 'time', 'step', 'valid_time']]
                u = ds_u[u_vars[0]]

                # Fetch V component
                ds_v = H.xarray(v_search, remove_grib=True)
                v_vars = [v for v in ds_v.data_vars if 'v' in v.lower() or 'vgrd' in v.lower()]
                if not v_vars:
                    v_vars = [v for v in ds_v.data_vars if v not in ['latitude', 'longitude', 'time', 'step', 'valid_time']]
                v = ds_v[v_vars[0]]

                # Subset to region if specified
                if region is not None:
                    u = self._subset_region(u, region)
                    v = self._subset_region(v, region)

                return u, v

            except Exception as e:
                logger.error(f"Error fetching wind components at {level}: {e}")
                raise

    def _convert_units(self, da: xr.DataArray, variable: Variable) -> xr.DataArray:
        """Apply unit conversions based on variable type."""

        units = da.attrs.get('units', '')

        # Temperature conversions
        if variable.units == "F" and units == "K":
            da = (da - 273.15) * 9/5 + 32
            da.attrs['units'] = 'F'
        elif variable.units == "C" and units == "K":
            da = da - 273.15
            da.attrs['units'] = 'C'

        # Pressure conversions
        elif variable.units == "mb" and units == "Pa":
            da = da / 100
            da.attrs['units'] = 'mb'

        # Height conversions (m to dm for 500mb heights)
        elif variable.units == "dm" and units == "m":
            da = da / 10
            da.attrs['units'] = 'dm'
        elif variable.units == "dm" and units == "gpm":
            da = da / 10
            da.attrs['units'] = 'dm'

        # Wind speed conversions (m/s to knots)
        elif variable.units == "kt" and units == "m/s":
            da = da * 1.94384
            da.attrs['units'] = 'kt'
        elif variable.units == "kt" and units == "m s**-1":
            da = da * 1.94384
            da.attrs['units'] = 'kt'

        # Precipitation conversions (kg/m² to inches)
        elif variable.units == "in" and units in ["kg/m^2", "kg m**-2", "kg/m²"]:
            da = da / 25.4  # mm to inches
            da.attrs['units'] = 'in'

        return da

    def _subset_region(self, da: xr.DataArray, region: Region) -> xr.DataArray:
        """Subset data to a geographic region."""
        west, east, south, north = region.bounds

        # Skip subsetting for global extent
        is_global_lon = (west <= -180 and east >= 180) or (east - west >= 360)
        is_global_lat = (south <= -90 and north >= 90)

        if is_global_lon and is_global_lat:
            return da

        # Handle coordinate names (could be lat/lon or latitude/longitude)
        lat_name = 'latitude' if 'latitude' in da.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in da.coords else 'lon'

        # Subset longitude if not global
        if not is_global_lon:
            # Handle crossing the dateline
            if west > east:
                # Region crosses dateline (e.g., Pacific)
                da_west = da.sel({lon_name: slice(west, 360)})
                da_east = da.sel({lon_name: slice(0, east)})
                da = xr.concat([da_west, da_east], dim=lon_name)
            else:
                # Convert negative longitudes to 0-360 if needed
                lons = da[lon_name].values
                if np.any(lons < 0):
                    # Data uses -180 to 180
                    da = da.sel({lon_name: slice(west, east)})
                else:
                    # Data uses 0 to 360
                    west_360 = west % 360 if west < 0 else west
                    east_360 = east % 360 if east < 0 else east
                    da = da.sel({lon_name: slice(west_360, east_360)})

        # Subset latitude if not global
        if not is_global_lat:
            # Check if latitude is descending (common in GFS: 90 to -90)
            lat_vals = da[lat_name].values
            if lat_vals[0] > lat_vals[-1]:
                # Descending latitude: swap bounds
                da = da.sel({lat_name: slice(north, south)})
            else:
                da = da.sel({lat_name: slice(south, north)})

        return da

    def check_data_availability(self, init_time: datetime, forecast_hour: int) -> bool:
        """Check if data is available for the given init time and forecast hour."""
        try:
            H = self._get_herbie(init_time, forecast_hour)
            # Try to get the inventory - if it works, data is available
            inv = H.inventory()
            return len(inv) > 0
        except Exception:
            return False

    def get_historical_analysis_times(
        self,
        days_back: int = 7,
        interval_hours: int = 6,
    ) -> List[Tuple[datetime, int]]:
        """
        Get list of (init_time, forecast_hour=0) pairs for historical analysis data.

        GFS F000 (analysis) from past init times provides historical data.
        This allows building a time series from the past to present.

        Args:
            days_back: Number of days of history to include
            interval_hours: Time interval between analysis times (6 = every 6 hours)

        Returns:
            List of (init_time, forecast_hour=0) tuples, oldest first
        """
        times = []
        latest = self.get_latest_init_time()

        # Go back in time
        hours_back = days_back * 24
        for h in range(hours_back, 0, -interval_hours):
            init_time = latest - timedelta(hours=h)
            # Snap to valid init hour
            while init_time.hour not in self.init_hours:
                init_time = init_time - timedelta(hours=1)
            times.append((init_time, 0))

        # Add current analysis
        times.append((latest, 0))

        return times

    def get_combined_time_sequence(
        self,
        historical_days: int = 7,
        forecast_hours_ahead: int = 72,
        interval_hours: int = 6,
    ) -> List[Tuple[datetime, int, datetime]]:
        """
        Get combined sequence of historical analyses + future forecasts.

        Returns tuples of (init_time, forecast_hour, valid_time) that can be
        used to build a continuous time series from past to future.

        Args:
            historical_days: Days of history to include (using F000 from past runs)
            forecast_hours_ahead: Forecast hours to include from latest run
            interval_hours: Time interval between data points

        Returns:
            List of (init_time, forecast_hour, valid_time) tuples, sorted by valid_time
        """
        sequence = []
        latest_init = self.get_latest_init_time()

        # Historical: Use F000 from past init times
        for h in range(historical_days * 24, 0, -interval_hours):
            past_init = latest_init - timedelta(hours=h)
            # Snap to valid init hour
            while past_init.hour not in self.init_hours:
                past_init = past_init - timedelta(hours=1)
            valid = self.valid_time(past_init, 0)
            sequence.append((past_init, 0, valid))

        # Current analysis (F000 from latest run)
        sequence.append((latest_init, 0, self.valid_time(latest_init, 0)))

        # Future: Use forecasts from latest run
        for fhr in range(interval_hours, forecast_hours_ahead + 1, interval_hours):
            valid = self.valid_time(latest_init, fhr)
            sequence.append((latest_init, fhr, valid))

        # Sort by valid time and remove duplicates
        sequence = sorted(set(sequence), key=lambda x: x[2])

        return sequence
