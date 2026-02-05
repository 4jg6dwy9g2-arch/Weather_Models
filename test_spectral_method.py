#!/usr/bin/env python3
"""
Simple validation test for the spectral decomposition method.
Creates synthetic wave patterns and verifies the spectral method correctly identifies them.
"""

import numpy as np
import xarray as xr
from rossby_waves import calculate_wave_number


def create_synthetic_wave_field(wave_number: int, amplitude: float = 200.0,
                                  n_lon: int = 144, mean_height: float = 5500.0):
    """
    Create a synthetic geopotential height field with a specified wave number.

    Args:
        wave_number: The wave number to create (number of complete waves around circle)
        amplitude: Wave amplitude in meters
        n_lon: Number of longitude points
        mean_height: Mean height in meters

    Returns:
        xarray DataArray with synthetic z500 field
    """
    # Create longitude coordinates (0 to 360)
    lons = np.linspace(0, 360, n_lon, endpoint=False)
    lats = np.array([55.0])  # Single latitude

    # Create wave pattern: Z = mean + amplitude * cos(k * longitude)
    lon_rad = np.deg2rad(lons)
    heights = mean_height + amplitude * np.cos(wave_number * lon_rad)

    # Create 2D field (1 latitude x n_lon longitudes)
    field_2d = heights[np.newaxis, :]

    # Create xarray DataArray
    data = xr.DataArray(
        field_2d,
        coords={'lat': lats, 'lon': lons},
        dims=['lat', 'lon'],
        name='z500'
    )

    return data


def test_single_wave():
    """Test that the spectral method correctly identifies a single wave number."""
    print("=" * 70)
    print("TEST 1: Single Wave Detection")
    print("=" * 70)

    for k in [3, 5, 7]:
        print(f"\nTesting wave number {k}...")
        field = create_synthetic_wave_field(wave_number=k, amplitude=200.0)
        result = calculate_wave_number(field, latitude=55.0)

        detected_wave = result['wave_number']
        detected_amp = result['top_3_amplitudes'][0] if result['top_3_amplitudes'] else None
        variance = result['top_3_variance'][0] if result['top_3_variance'] else None

        print(f"  Input: wave {k}, amplitude 200m")
        print(f"  Detected: wave {detected_wave}, amplitude {detected_amp:.1f}m, variance {variance:.1f}%")

        # Validation
        if detected_wave == k:
            print(f"  ✓ PASS: Correctly identified wave {k}")
        else:
            print(f"  ✗ FAIL: Expected wave {k}, got {detected_wave}")

        if detected_amp is not None and abs(detected_amp - 200.0) < 10.0:
            print(f"  ✓ PASS: Amplitude within 10m of expected")
        else:
            print(f"  ⚠ WARNING: Amplitude {detected_amp:.1f}m differs from expected 200m")


def test_multiple_waves():
    """Test detection of multiple superposed waves."""
    print("\n" + "=" * 70)
    print("TEST 2: Multiple Wave Detection")
    print("=" * 70)

    # Create field with wave 3 (amplitude 250m) + wave 6 (amplitude 150m)
    print("\nCreating synthetic field: wave 3 (250m) + wave 6 (150m)...")

    n_lon = 144
    lons = np.linspace(0, 360, n_lon, endpoint=False)
    lats = np.array([55.0])
    lon_rad = np.deg2rad(lons)

    mean_height = 5500.0
    heights = mean_height + 250.0 * np.cos(3 * lon_rad) + 150.0 * np.cos(6 * lon_rad)
    field_2d = heights[np.newaxis, :]

    field = xr.DataArray(
        field_2d,
        coords={'lat': lats, 'lon': lons},
        dims=['lat', 'lon'],
        name='z500'
    )

    result = calculate_wave_number(field, latitude=55.0)

    print(f"\nDetected dominant wave numbers: {result['dominant_wave_numbers']}")
    print(f"Amplitudes: {[f'{a:.1f}m' for a in result['top_3_amplitudes']]}")
    print(f"Variance explained: {[f'{v:.1f}%' for v in result['top_3_variance']]}")

    # Check if wave 3 and 6 are in top detections
    if 3 in result['dominant_wave_numbers'] and 6 in result['dominant_wave_numbers']:
        print("✓ PASS: Both waves 3 and 6 detected")
    else:
        print("✗ FAIL: Expected to detect waves 3 and 6")


def test_variance_conservation():
    """Test that variance explained by all waves sums to ~100%."""
    print("\n" + "=" * 70)
    print("TEST 3: Variance Conservation")
    print("=" * 70)

    # Create random-ish field with multiple waves
    print("\nCreating complex multi-wave field...")
    field = create_synthetic_wave_field(wave_number=4, amplitude=180.0)

    # Add some additional waves
    n_lon = 144
    lons = np.linspace(0, 360, n_lon, endpoint=False)
    lon_rad = np.deg2rad(lons)
    additional = 120.0 * np.cos(6 * lon_rad) + 80.0 * np.cos(2 * lon_rad)
    field.values[0, :] += additional

    result = calculate_wave_number(field, latitude=55.0, max_wave_number=15)

    # Sum variance explained across all wave numbers
    total_variance_explained = sum(result['variance_explained'].values())

    print(f"\nTotal variance in field: {result['total_variance']:.1f} m²")
    print(f"Sum of variance explained (waves 1-15): {total_variance_explained:.1f}%")
    print(f"Top 3 waves explain: {sum(result['top_3_variance']):.1f}%")

    if 90 <= total_variance_explained <= 110:
        print("✓ PASS: Variance explained close to 100%")
    else:
        print("⚠ WARNING: Variance explained should be close to 100%")


def test_real_world_range():
    """Test with realistic atmospheric values."""
    print("\n" + "=" * 70)
    print("TEST 4: Realistic Atmospheric Values")
    print("=" * 70)

    # Typical NH winter pattern: wave 4-5 dominance
    print("\nSimulating typical NH winter pattern (wave 5, 300m amplitude)...")
    field = create_synthetic_wave_field(wave_number=5, amplitude=300.0, mean_height=5400.0)

    result = calculate_wave_number(field, latitude=55.0)

    print(f"Detected wave number: {result['wave_number']}")
    print(f"Amplitude: {result['top_3_amplitudes'][0]:.1f}m")
    print(f"Method: {result['method']}")
    print(f"Latitude: {result['latitude_used']}°N")

    # Check values are in realistic ranges
    checks = [
        (result['wave_number'] == 5, "Wave number correct"),
        (250 < result['top_3_amplitudes'][0] < 350, "Amplitude in realistic range"),
        (result['method'] == 'spectral_decomposition', "Using spectral method"),
    ]

    for passed, description in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {description}")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("SPECTRAL METHOD VALIDATION TESTS")
    print("=" * 70)

    test_single_wave()
    test_multiple_waves()
    test_variance_conservation()
    test_real_world_range()

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("\nIf all tests pass, the spectral decomposition method is working correctly.")
    print("The method should correctly identify wave numbers, amplitudes, and variance.")
    print()
