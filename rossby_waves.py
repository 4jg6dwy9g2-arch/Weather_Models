"""
Rossby Wave Number Analysis Module

Calculates the number of Rossby waves in the Northern Hemisphere from 500 hPa
geopotential height fields. Lower wave numbers (2-4) indicate larger, more
persistent patterns with better forecast reliability. Higher wave numbers (5-8+)
suggest faster-evolving, more chaotic patterns with reduced predictability.
"""

from typing import Dict, List, Optional, Union
import logging

import numpy as np
import xarray as xr
from scipy import signal
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)


def calculate_wave_number(
    z500_field: xr.DataArray,
    latitude: float = 55.0,
    smoothing_width: int = 5,
    prominence_threshold: float = 5.0,
) -> Dict[str, Union[float, int, List[int], str]]:
    """
    Calculate Rossby wave number from 500 hPa geopotential height field.

    Args:
        z500_field: xarray DataArray containing 500 hPa geopotential heights (in dm)
        latitude: Latitude circle to analyze (default 55.0°N)
        smoothing_width: Number of points for running mean smoothing (default 5)
        prominence_threshold: Minimum prominence for peak detection in dm (default 10.0)

    Returns:
        Dictionary containing:
            - wave_number: Average number of waves (ridges + troughs) / 2
            - ridges: Number of ridges (high pressure centers)
            - troughs: Number of troughs (low pressure centers)
            - dominant_wave_numbers: List of dominant wave numbers from FFT
            - latitude_used: Latitude circle used for analysis
            - method: Method used for calculation
    """
    try:
        # Determine coordinate names
        lat_name = 'latitude' if 'latitude' in z500_field.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in z500_field.coords else 'lon'

        # Extract data along the specified latitude circle
        # Find the closest available latitude
        lats = z500_field[lat_name].values
        lat_idx = np.argmin(np.abs(lats - latitude))
        actual_latitude = float(lats[lat_idx])

        logger.info(f"Extracting z500 along {actual_latitude:.1f}°N (requested {latitude}°N)")

        # Extract the height values along this latitude circle
        z500_slice = z500_field.isel({lat_name: lat_idx})

        # Ensure we have longitude as the dimension
        if lon_name not in z500_slice.dims:
            logger.error(f"Longitude dimension '{lon_name}' not found in z500 slice")
            return _empty_result(actual_latitude)

        # Sort by longitude to ensure proper ordering
        z500_slice = z500_slice.sortby(lon_name)
        heights = z500_slice.values

        # Remove any NaN values
        if np.any(np.isnan(heights)):
            logger.warning(f"Found {np.sum(np.isnan(heights))} NaN values in z500 field, interpolating")
            # Simple linear interpolation for NaNs
            nans = np.isnan(heights)
            if np.all(nans):
                logger.error("All values are NaN in z500 field")
                return _empty_result(actual_latitude)
            x = np.arange(len(heights))
            heights[nans] = np.interp(x[nans], x[~nans], heights[~nans])

        # Apply smoothing to remove small-scale noise
        if smoothing_width > 1:
            # Wrap the data at the boundaries (periodic for full latitude circle)
            heights_padded = np.concatenate([
                heights[-smoothing_width:],
                heights,
                heights[:smoothing_width]
            ])
            heights_smoothed = uniform_filter1d(heights_padded, smoothing_width, mode='constant')
            # Extract the middle portion (remove padding)
            heights = heights_smoothed[smoothing_width:-smoothing_width]

        # Find ridges (local maxima) and troughs (local minima)
        ridges, ridge_properties = signal.find_peaks(
            heights,
            prominence=prominence_threshold,
            distance=3  # Minimum separation between peaks (in grid points)
        )

        troughs, trough_properties = signal.find_peaks(
            -heights,  # Invert to find minima
            prominence=prominence_threshold,
            distance=3
        )

        num_ridges = len(ridges)
        num_troughs = len(troughs)

        logger.info(f"Found {num_ridges} ridges and {num_troughs} troughs at {actual_latitude:.1f}°N")

        # Calculate wave number as average of ridges and troughs
        wave_number = (num_ridges + num_troughs) / 2.0

        # Perform FFT to identify dominant wave numbers
        dominant_waves = _fft_wave_analysis(heights)

        return {
            'wave_number': round(wave_number, 1),
            'ridges': num_ridges,
            'troughs': num_troughs,
            'dominant_wave_numbers': dominant_waves,
            'latitude_used': round(actual_latitude, 1),
            'method': 'peak_detection'
        }

    except Exception as e:
        logger.error(f"Error calculating wave number: {e}")
        return _empty_result(latitude)


def _fft_wave_analysis(heights: np.ndarray, max_waves: int = 10) -> List[int]:
    """
    Use FFT to identify dominant wave numbers in the height field.

    Args:
        heights: 1D array of geopotential heights along a latitude circle
        max_waves: Maximum wave number to consider (default 10)

    Returns:
        List of dominant wave numbers, sorted by amplitude (strongest first)
    """
    try:
        # Detrend the data (remove mean)
        heights_detrended = heights - np.mean(heights)

        # Compute FFT
        fft = np.fft.fft(heights_detrended)
        power = np.abs(fft[:len(fft)//2])**2  # Power spectrum (positive frequencies only)

        # Wave numbers correspond to FFT frequency bins
        # wave_number k means k complete waves around the latitude circle
        n = len(heights)
        wave_numbers = np.arange(0, len(power))

        # Focus on synoptic-scale wave numbers (2-10)
        valid_range = (wave_numbers >= 2) & (wave_numbers <= max_waves)
        valid_wave_numbers = wave_numbers[valid_range]
        valid_power = power[valid_range]

        # Sort by power (strongest waves first)
        sorted_indices = np.argsort(valid_power)[::-1]

        # Return top 3 dominant wave numbers
        dominant = [int(valid_wave_numbers[i]) for i in sorted_indices[:3]]

        logger.debug(f"FFT dominant wave numbers: {dominant}")

        return dominant

    except Exception as e:
        logger.warning(f"FFT wave analysis failed: {e}")
        return []


def _empty_result(latitude: float) -> Dict[str, Union[float, int, List[int], str, None]]:
    """Return an empty result dictionary when calculation fails."""
    return {
        'wave_number': None,
        'ridges': 0,
        'troughs': 0,
        'dominant_wave_numbers': [],
        'latitude_used': round(latitude, 1),
        'method': 'failed'
    }


def calculate_wave_metrics_multi_latitude(
    z500_field: xr.DataArray,
    latitudes: List[float] = [45.0, 50.0, 55.0, 60.0],
) -> Dict[str, Union[float, Dict[float, Dict]]]:
    """
    Calculate wave numbers at multiple latitudes and provide statistics.

    This can be useful for understanding meridional variations in wave patterns.

    Args:
        z500_field: xarray DataArray containing 500 hPa geopotential heights
        latitudes: List of latitudes to analyze (default [45, 50, 55, 60]°N)

    Returns:
        Dictionary containing:
            - mean_wave_number: Average wave number across all latitudes
            - std_wave_number: Standard deviation of wave numbers
            - by_latitude: Dict mapping each latitude to its wave metrics
    """
    results_by_lat = {}
    wave_numbers = []

    for lat in latitudes:
        result = calculate_wave_number(z500_field, latitude=lat)
        results_by_lat[lat] = result

        if result['wave_number'] is not None:
            wave_numbers.append(result['wave_number'])

    if wave_numbers:
        mean_wave = np.mean(wave_numbers)
        std_wave = np.std(wave_numbers)
    else:
        mean_wave = None
        std_wave = None

    return {
        'mean_wave_number': round(mean_wave, 1) if mean_wave is not None else None,
        'std_wave_number': round(std_wave, 1) if std_wave is not None else None,
        'by_latitude': results_by_lat
    }
