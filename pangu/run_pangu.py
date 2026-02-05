#!/usr/bin/env python3
"""
PanguWeather AI Weather Model Runner

This script runs the PanguWeather model from Huawei to generate weather forecasts.
PanguWeather works on all platforms including Apple Silicon!
"""

import subprocess
import sys
from datetime import datetime, timedelta

def run_panguweather(date=None, time="00", lead_time=24, output_path="./forecasts"):
    """
    Run PanguWeather forecast.

    Args:
        date: Date string (YYYY-MM-DD) or None for yesterday
        time: Time string (00, 06, 12, 18)
        lead_time: Forecast length in hours
        output_path: Where to save forecast outputs
    """

    print("=" * 60)
    print("PanguWeather AI Weather Model")
    print("=" * 60)
    print()
    print("About PanguWeather:")
    print("- Developed by Huawei Research")
    print("- Published in Nature 2023")
    print("- Fast, accurate AI weather forecasting")
    print("- Works on all platforms (CPU, NVIDIA GPU, Apple Silicon)")
    print()

    # Use yesterday if no date specified
    if date is None:
        yesterday = datetime.now() - timedelta(days=1)
        date = yesterday.strftime("%Y%m%d")  # Format as YYYYMMDD
    else:
        # Convert YYYY-MM-DD to YYYYMMDD if needed
        date = date.replace("-", "")

    print(f"Configuration:")
    print(f"  Initial date: {date}")
    print(f"  Initial time: {time}:00 UTC")
    print(f"  Forecast length: {lead_time} hours")
    print(f"  Output path: {output_path}")
    print()

    # Build command
    cmd = [
        "ai-models",
        "panguweather",
        "--input", "ecmwf-open-data",
        "--date", date,
        "--time", time,
        "--lead-time", str(lead_time),
        "--path", output_path,
    ]

    print("Running PanguWeather...")
    print(f"Command: {' '.join(cmd)}")
    print()
    print("-" * 60)

    try:
        # Run the model
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)

        print("-" * 60)
        print()
        print("✅ Forecast complete!")
        print(f"Output saved to: {output_path}")
        print()
        print("The forecast is saved in GRIB format.")
        print("You can view it with tools like:")
        print("  - xarray/cfgrib in Python")
        print("  - Panoply (NASA)")
        print("  - ncview")

        return True

    except subprocess.CalledProcessError as e:
        print()
        print("❌ Error running PanguWeather:")
        print(f"   {e}")
        return False
    except KeyboardInterrupt:
        print()
        print("❌ Cancelled by user")
        return False


def main():
    """Main execution."""
    print()

    # Parse simple command line args
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("Usage: python run_pangu.py [DATE] [TIME] [LEAD_TIME]")
        print()
        print("Arguments:")
        print("  DATE       : Date in YYYY-MM-DD format (default: yesterday)")
        print("  TIME       : Hour 00, 06, 12, or 18 (default: 00)")
        print("  LEAD_TIME  : Forecast hours (default: 24)")
        print()
        print("Examples:")
        print("  python run_pangu.py")
        print("  python run_pangu.py 2026-01-28 12 48")
        sys.exit(0)

    # Get arguments or use defaults
    date = sys.argv[1] if len(sys.argv) > 1 else None
    time = sys.argv[2] if len(sys.argv) > 2 else "00"
    lead_time = int(sys.argv[3]) if len(sys.argv) > 3 else 24

    # Run the forecast
    success = run_panguweather(date=date, time=time, lead_time=lead_time)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
