import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import logging

from gfs import GFSModel
from ecmwf_aifs import AIFSModel, AIFS_VARIABLES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Define the variables we want to plot for GFS
TEMP_2M_GFS = Variable(
    name="t2m",
    display_name="2m Temperature",
    units="F",
    herbie_search=":TMP:2 m above ground:",
    category="surface",
    colormap="RdYlBu_r",
    contour_levels=list(range(-40, 120, 5))
)

PRECIP_GFS = Variable(
    name="precip",
    display_name="Total Precipitation",
    units="in",
    herbie_search=":APCP:",
    category="surface",
    colormap="precip",
    contour_levels=[0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
)

# Define the location of Fairfax, VA
# Coordinates: 38.8462 N, 77.3064 W
# Bounds aligned to 0.25 degree grid for GFS compatibility
FAIRFAX_VA = Region("Fairfax, VA", (-77.5, -77.0, 38.5, 39.0))

def fetch_gfs_data(model, init_time, forecast_hours, region):
    """Fetch temperature and precipitation data from GFS."""
    temperatures = []
    precipitations = []
    valid_times = []

    for hour in forecast_hours:
        logger.info(f"GFS: Fetching data for F{hour:03d}")

        # Fetch temperature
        try:
            temp_data = model.fetch_data(TEMP_2M_GFS, init_time, hour, region)
            temp_val = float(temp_data.values.mean())
            temperatures.append(temp_val)
            logger.info(f"GFS Temperature at F{hour:03d}: {temp_val:.1f} F")
        except Exception as e:
            logger.error(f"GFS: Could not fetch temperature at F{hour:03d}: {e}")
            temperatures.append(np.nan)

        # Fetch precipitation
        try:
            precip_data = model.fetch_data(PRECIP_GFS, init_time, hour, region)
            precip_val = float(precip_data.values.mean())
            precipitations.append(precip_val)
            logger.info(f"GFS Precipitation at F{hour:03d}: {precip_val:.3f} in")
        except Exception as e:
            logger.error(f"GFS: Could not fetch precipitation at F{hour:03d}: {e}")
            precipitations.append(np.nan)

        valid_times.append(init_time + np.timedelta64(hour, 'h'))

    return temperatures, precipitations, valid_times

def fetch_aifs_data(model, init_time, forecast_hours, region):
    """Fetch temperature and precipitation data from ECMWF AIFS."""
    temperatures = []
    precipitations = []
    valid_times = []

    # AIFS variables
    temp_var = AIFS_VARIABLES["t2m"]
    precip_var = AIFS_VARIABLES["tp"]

    for hour in forecast_hours:
        logger.info(f"AIFS: Fetching data for F{hour:03d}")

        # Fetch temperature
        try:
            temp_data = model.fetch_data(temp_var, init_time, hour, region)
            temp_val = float(temp_data.values.mean())
            temperatures.append(temp_val)
            logger.info(f"AIFS Temperature at F{hour:03d}: {temp_val:.1f} F")
        except Exception as e:
            logger.error(f"AIFS: Could not fetch temperature at F{hour:03d}: {e}")
            temperatures.append(np.nan)

        # Fetch precipitation
        try:
            precip_data = model.fetch_data(precip_var, init_time, hour, region)
            precip_val = float(precip_data.values.mean())
            precipitations.append(precip_val)
            logger.info(f"AIFS Precipitation at F{hour:03d}: {precip_val:.3f} in")
        except Exception as e:
            logger.error(f"AIFS: Could not fetch precipitation at F{hour:03d}: {e}")
            precipitations.append(np.nan)

        valid_times.append(init_time + np.timedelta64(hour, 'h'))

    return temperatures, precipitations, valid_times

def plot_weather():
    """
    Fetches and plots temperature and precipitation for Fairfax, VA
    from both GFS and ECMWF AIFS models.
    """
    # Initialize models
    gfs_model = GFSModel()
    aifs_model = AIFSModel()

    # Get the latest model runs
    gfs_init_time = gfs_model.get_latest_init_time()
    aifs_init_time = aifs_model.get_latest_init_time()

    logger.info(f"Latest GFS run: {gfs_init_time}")
    logger.info(f"Latest AIFS run: {aifs_init_time}")

    # Define the forecast hours to plot (both models use 6-hour intervals)
    # GFS goes to 384 hours, AIFS goes to 360 hours
    # Use 360 hours (15 days) - the common maximum for both models
    forecast_hours = list(range(0, 361, 6))

    # Fetch data from both models
    logger.info("Fetching GFS data...")
    gfs_temps, gfs_precips, gfs_times = fetch_gfs_data(
        gfs_model, gfs_init_time, forecast_hours, FAIRFAX_VA
    )

    logger.info("Fetching AIFS data...")
    aifs_temps, aifs_precips, aifs_times = fetch_aifs_data(
        aifs_model, aifs_init_time, forecast_hours, FAIRFAX_VA
    )

    # Create the plot with 2 subplots (temperature and precipitation)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot temperature comparison
    ax1.plot(gfs_times, gfs_temps, color='tab:red', marker='o', linestyle='-',
             label=f'GFS ({gfs_init_time.strftime("%Y-%m-%d %HZ")})', markersize=4)
    ax1.plot(aifs_times, aifs_temps, color='tab:orange', marker='s', linestyle='--',
             label=f'AIFS ({aifs_init_time.strftime("%Y-%m-%d %HZ")})', markersize=4)
    ax1.set_ylabel('Temperature (F)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Temperature Forecast Comparison')

    # Plot precipitation comparison
    ax2.plot(gfs_times, gfs_precips, color='tab:blue', marker='o', linestyle='-',
             label=f'GFS ({gfs_init_time.strftime("%Y-%m-%d %HZ")})', markersize=4)
    ax2.plot(aifs_times, aifs_precips, color='tab:cyan', marker='s', linestyle='--',
             label=f'AIFS ({aifs_init_time.strftime("%Y-%m-%d %HZ")})', markersize=4)
    ax2.set_xlabel('Valid Time')
    ax2.set_ylabel('Precipitation (in)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Precipitation Forecast Comparison')

    # Rotate x-axis labels for readability
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Overall title
    fig.suptitle(f"Weather Forecast for Fairfax, VA\nGFS vs ECMWF AIFS Comparison",
                 fontsize=14, fontweight='bold')

    fig.tight_layout()

    # Save the plot
    output_file = "fairfax_weather_forecast.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved to {output_file}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    gfs_temps_valid = [t for t in gfs_temps if not np.isnan(t)]
    aifs_temps_valid = [t for t in aifs_temps if not np.isnan(t)]

    if gfs_temps_valid:
        print(f"GFS Temperature range: {min(gfs_temps_valid):.1f}F - {max(gfs_temps_valid):.1f}F")
    if aifs_temps_valid:
        print(f"AIFS Temperature range: {min(aifs_temps_valid):.1f}F - {max(aifs_temps_valid):.1f}F")

    gfs_precips_valid = [p for p in gfs_precips if not np.isnan(p)]
    aifs_precips_valid = [p for p in aifs_precips if not np.isnan(p)]

    if gfs_precips_valid:
        print(f"GFS Total Precipitation: {sum(gfs_precips_valid):.2f} in")
    if aifs_precips_valid:
        print(f"AIFS Total Precipitation: {sum(aifs_precips_valid):.2f} in")

    print("="*60)

if __name__ == "__main__":
    plot_weather()
