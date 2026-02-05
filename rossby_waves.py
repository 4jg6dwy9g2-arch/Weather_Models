"""
Rossby Wave Number Analysis Module

Calculates the number of Rossby waves in the Northern Hemisphere from 500 hPa
geopotential height fields using spectral decomposition method (Blackmon et al., 1984;
Branstator, 1987). Lower wave numbers (2-4) indicate larger, more persistent patterns
with better forecast reliability. Higher wave numbers (5-8+) suggest faster-evolving,
more chaotic patterns with reduced predictability.

References:
- Blackmon, M. L., et al. (1984): An observational study of the Northern Hemisphere
  wintertime circulation. J. Atmos. Sci., 41, 1365-1383.
- Branstator, G. (1987): A striking example of the atmosphere's leading traveling pattern.
  J. Atmos. Sci., 44, 2310-2323.
- Kornhuber, K., et al. (2017): Evidence for wave resonance as a key mechanism for
  generating high-amplitude quasi-stationary waves in boreal summer. Clim. Dyn., 49, 1961-1979.
"""

from typing import Dict, List, Optional, Union, Tuple
import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def calculate_wave_number(
    z500_field: xr.DataArray,
    latitude: float = 55.0,
    max_wave_number: int = 15,
    min_wave_number: int = 1,
) -> Dict[str, Union[float, int, List, str, Dict]]:
    """
    Calculate Rossby wave number from 500 hPa geopotential height field using
    spectral decomposition method.

    This method computes geopotential height anomalies (deviation from zonal mean)
    and performs Fourier decomposition to identify dominant wave numbers based on
    power spectrum analysis (Blackmon et al., 1984; Branstator, 1987).

    Args:
        z500_field: xarray DataArray containing 500 hPa geopotential heights (in meters or dm)
        latitude: Latitude circle to analyze (default 55.0°N)
        max_wave_number: Maximum wave number to analyze (default 15)
        min_wave_number: Minimum wave number to analyze (default 1)

    Returns:
        Dictionary containing:
            - wave_number: Dominant wave number (highest power)
            - dominant_wave_numbers: List of top 3 wave numbers by power
            - wave_amplitudes: Dict mapping wave number to amplitude (in meters)
            - total_variance: Total variance of anomaly field
            - variance_explained: Dict mapping wave number to % variance explained
            - latitude_used: Latitude circle used for analysis
            - method: Method used for calculation ('spectral_decomposition')
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

        # Convert to meters if in decameters (typical range for z500 is 4800-6000m)
        if np.mean(heights) < 1000:  # Likely in decameters
            heights = heights * 10.0
            logger.debug("Converted heights from decameters to meters")

        # Calculate zonal mean (average around the latitude circle)
        zonal_mean = np.mean(heights)

        # Calculate anomalies (deviation from zonal mean)
        anomalies = heights - zonal_mean

        logger.info(f"Zonal mean: {zonal_mean:.1f} m, Anomaly range: [{np.min(anomalies):.1f}, {np.max(anomalies):.1f}] m")

        # Perform spectral decomposition
        wave_analysis = _spectral_decomposition(
            anomalies,
            max_wave_number=max_wave_number,
            min_wave_number=min_wave_number
        )

        # Add metadata
        wave_analysis['latitude_used'] = round(actual_latitude, 1)
        wave_analysis['method'] = 'spectral_decomposition'

        return wave_analysis

    except Exception as e:
        logger.error(f"Error calculating wave number: {e}")
        return _empty_result(latitude)


def _spectral_decomposition(
    anomalies: np.ndarray,
    max_wave_number: int = 15,
    min_wave_number: int = 1,
) -> Dict[str, Union[float, int, List, Dict]]:
    """
    Perform spectral decomposition of geopotential height anomalies using Fourier analysis.

    Following Blackmon et al. (1984) and Branstator (1987), this decomposes the anomaly
    field into contributions from different zonal wave numbers and identifies dominant modes.

    Args:
        anomalies: 1D array of geopotential height anomalies along a latitude circle (meters)
        max_wave_number: Maximum wave number to analyze (default 15)
        min_wave_number: Minimum wave number to analyze (default 1)

    Returns:
        Dictionary containing wave analysis results
    """
    try:
        n_points = len(anomalies)

        # Compute FFT of anomalies
        fft_coeffs = np.fft.fft(anomalies)

        # Calculate power spectrum for each wave number
        # For a real signal Z = A*cos(k*lon), FFT[k] = n*A/2
        # Power spectrum (one-sided, accounting for both +k and -k frequencies):
        # - For k=0 (zonal mean, should be ~0 after removing mean): |FFT[0]|^2 / n^2
        # - For k>0: 2 * |FFT[k]|^2 / n^2 (factor of 2 accounts for negative frequency)
        power_spectrum_raw = np.abs(fft_coeffs[:n_points//2])**2 / (n_points**2)

        # Account for negative frequencies (double power for k>0)
        power_spectrum = power_spectrum_raw.copy()
        power_spectrum[1:] *= 2  # Double all except k=0 (zonal mean)

        # Total variance of the anomaly field
        total_variance = np.var(anomalies)

        # Wave numbers correspond to FFT indices
        # wave number k = number of complete sinusoidal cycles around latitude circle
        wave_numbers = np.arange(0, len(power_spectrum))

        # Calculate amplitude for each wave number (in meters)
        # For wave k: Z = A*cos(k*lon), variance = A^2/2, power = A^2/4 * 2 = A^2/2
        # Therefore: A = sqrt(2 * power)
        # But we need to account for the FFT normalization:
        # |FFT[k]| = n*A/2, so A = 2*|FFT[k]|/n
        amplitudes = np.sqrt(2 * power_spectrum)

        # Focus on synoptic-scale wave numbers (min to max)
        valid_range = (wave_numbers >= min_wave_number) & (wave_numbers <= max_wave_number)
        valid_wave_numbers = wave_numbers[valid_range]
        valid_power = power_spectrum[valid_range]
        valid_amplitudes = amplitudes[valid_range]

        # Calculate variance explained by each wave number
        # Variance explained = power / total_variance * 100
        if total_variance > 0:
            variance_explained = (valid_power / total_variance) * 100
        else:
            variance_explained = np.zeros_like(valid_power)

        # Sort by power (strongest waves first)
        sorted_indices = np.argsort(valid_power)[::-1]

        # Get top 3 dominant wave numbers
        top_n = min(3, len(sorted_indices))
        dominant_wave_numbers = [int(valid_wave_numbers[sorted_indices[i]]) for i in range(top_n)]
        dominant_amplitudes = [float(valid_amplitudes[sorted_indices[i]]) for i in range(top_n)]
        dominant_variance = [float(variance_explained[sorted_indices[i]]) for i in range(top_n)]

        # Primary wave number (highest power)
        primary_wave_number = dominant_wave_numbers[0] if dominant_wave_numbers else None

        # Create dictionaries for all wave numbers
        wave_amplitude_dict = {
            int(k): round(float(amp), 1)
            for k, amp in zip(valid_wave_numbers, valid_amplitudes)
        }

        variance_explained_dict = {
            int(k): round(float(var), 1)
            for k, var in zip(valid_wave_numbers, variance_explained)
        }

        logger.info(
            f"Dominant wave numbers: {dominant_wave_numbers} "
            f"(amplitudes: {[round(a, 1) for a in dominant_amplitudes]} m, "
            f"variance: {[round(v, 1) for v in dominant_variance]}%)"
        )

        return {
            'wave_number': primary_wave_number,
            'dominant_wave_numbers': dominant_wave_numbers,
            'wave_amplitudes': wave_amplitude_dict,
            'total_variance': round(float(total_variance), 1),
            'variance_explained': variance_explained_dict,
            'top_3_amplitudes': [round(float(a), 1) for a in dominant_amplitudes],
            'top_3_variance': [round(float(v), 1) for v in dominant_variance],
        }

    except Exception as e:
        logger.error(f"Spectral decomposition failed: {e}")
        return {
            'wave_number': None,
            'dominant_wave_numbers': [],
            'wave_amplitudes': {},
            'total_variance': 0.0,
            'variance_explained': {},
            'top_3_amplitudes': [],
            'top_3_variance': [],
        }


def _empty_result(latitude: float) -> Dict[str, Union[float, int, List, str, Dict, None]]:
    """Return an empty result dictionary when calculation fails."""
    return {
        'wave_number': None,
        'dominant_wave_numbers': [],
        'wave_amplitudes': {},
        'total_variance': 0.0,
        'variance_explained': {},
        'top_3_amplitudes': [],
        'top_3_variance': [],
        'latitude_used': round(latitude, 1),
        'method': 'failed'
    }


def calculate_wave_metrics_multi_latitude(
    z500_field: xr.DataArray,
    latitudes: List[float] = [45.0, 50.0, 55.0, 60.0],
    max_wave_number: int = 15,
) -> Dict[str, Union[float, Dict[float, Dict]]]:
    """
    Calculate wave numbers at multiple latitudes and provide statistics.

    This can be useful for understanding meridional variations in wave patterns
    and assessing the vertical structure of Rossby waves.

    Args:
        z500_field: xarray DataArray containing 500 hPa geopotential heights
        latitudes: List of latitudes to analyze (default [45, 50, 55, 60]°N)
        max_wave_number: Maximum wave number to analyze (default 15)

    Returns:
        Dictionary containing:
            - mean_wave_number: Average dominant wave number across all latitudes
            - std_wave_number: Standard deviation of wave numbers
            - mean_amplitude: Average wave amplitude across latitudes (meters)
            - by_latitude: Dict mapping each latitude to its wave metrics
    """
    results_by_lat = {}
    wave_numbers = []
    amplitudes = []

    for lat in latitudes:
        result = calculate_wave_number(z500_field, latitude=lat, max_wave_number=max_wave_number)
        results_by_lat[lat] = result

        if result['wave_number'] is not None:
            wave_numbers.append(result['wave_number'])
            # Get amplitude of dominant wave
            if result['top_3_amplitudes']:
                amplitudes.append(result['top_3_amplitudes'][0])

    if wave_numbers:
        mean_wave = np.mean(wave_numbers)
        std_wave = np.std(wave_numbers)
    else:
        mean_wave = None
        std_wave = None

    if amplitudes:
        mean_amp = np.mean(amplitudes)
    else:
        mean_amp = None

    return {
        'mean_wave_number': round(mean_wave, 1) if mean_wave is not None else None,
        'std_wave_number': round(std_wave, 1) if std_wave is not None else None,
        'mean_amplitude': round(mean_amp, 1) if mean_amp is not None else None,
        'by_latitude': results_by_lat
    }
