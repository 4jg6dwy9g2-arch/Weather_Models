# PanguWeather Flask App

A web interface for running and visualizing PanguWeather AI weather forecasts.

## Features

- **Run Forecasts**: Configure and run PanguWeather forecasts with custom dates, times, and lead times
- **Visual Dashboard**: View forecast time series with interactive charts
- **Run History**: Browse all saved forecast runs
- **Real-time Logs**: Watch model execution in real-time via streaming logs
- **Modern UI**: Clean, responsive interface similar to the main Weather Models app

## Installation

### Prerequisites

Make sure you have the PanguWeather model files already downloaded (the ONNX files in your current directory).

### Install Dependencies

```bash
pip install flask xarray cfgrib eccodes
```

If you encounter issues installing `eccodes`, you may need to install it via your system package manager first:

```bash
# macOS
brew install eccodes

# Linux (Ubuntu/Debian)
sudo apt-get install libeccodes-dev
```

## Usage

### Start the Flask App

```bash
cd /Users/kennypratt/Documents/ML_Weather_Models
python pangu_app.py
```

The app will start on http://localhost:5002

### Run a Forecast

1. Navigate to http://localhost:5002/run
2. Select initial date (typically yesterday, as ECMWF data has 1-2 day lag)
3. Choose initial time (00, 06, 12, or 18 UTC)
4. Set lead time (6-168 hours, in 6-hour increments)
5. Click "Run Forecast"
6. Watch the real-time logs as the model runs
7. View results on the dashboard

### View Forecasts

1. Navigate to http://localhost:5002 (Dashboard)
2. View statistics and latest forecast chart
3. Click on any saved run to view its data
4. Charts show:
   - Temperature at 850 hPa (lower atmosphere)
   - Temperature at 500 hPa (mid atmosphere)
   - Wind speed at 850 hPa

## Configuration

### Forecast Parameters

- **Initial Date**: Date for initial conditions (YYYY-MM-DD)
- **Initial Time**: Hour for initial conditions (00, 06, 12, or 18 UTC)
- **Lead Time**: Forecast length in hours (6-168, multiples of 6)

### Data Sources

- **Initial Conditions**: ECMWF Open Data (downloaded automatically)
- **Model Files**: PanguWeather ONNX files (must be pre-downloaded)

## Output

Forecasts are saved in two locations:

1. **GRIB Files**: `pangu_forecasts/` directory
   - Full spatial fields in GRIB2 format
   - Can be opened with xarray/cfgrib

2. **JSON Database**: `pangu_runs.json`
   - Run metadata and time series data
   - Used by the web interface

## File Structure

```
ML_Weather_Models/
├── pangu_app.py              # Flask application
├── pangu_runs.json            # Forecast runs database
├── pangu_forecasts/           # GRIB output files
├── templates/
│   ├── pangu_base.html        # Base template
│   ├── pangu_dashboard.html   # Dashboard page
│   └── pangu_run.html         # Run forecast page
├── pangu_weather_24.onnx      # Model file (24hr)
└── pangu_weather_6.onnx       # Model file (6hr)
```

## PanguWeather Model Details

- **Resolution**: 0.25° (~25km)
- **Pressure Levels**: 13 levels (1000-50 hPa)
- **Time Steps**: 6-hour increments
- **Max Forecast**: 7 days (168 hours)
- **Speed**: ~7-8 minutes for 24-hour forecast on Apple Silicon CPU

### Variables Available

- Temperature (t)
- Wind U/V components (u, v)
- Specific humidity (q)
- Geopotential height (z/gh)

## Tips

1. **Use Recent Dates**: ECMWF open data is typically available with 1-2 day lag
2. **Start with 24hrs**: Test with a 24-hour forecast first (faster)
3. **Check Logs**: Monitor the real-time logs to see progress
4. **Multiple Runs**: You can compare different initialization times
5. **Keep History**: The app stores up to 50 recent runs

## Troubleshooting

### "No such file or directory: './pangu_weather_24.onnx'"

The model files need to be in the working directory. Make sure you have:
- `pangu_weather_24.onnx`
- `pangu_weather_6.onnx`

These should have been downloaded when you first ran the model.

### GRIB File Reading Errors

Install cfgrib and eccodes:
```bash
brew install eccodes  # macOS
pip install cfgrib
```

### Port Already in Use

Change the port in `pangu_app.py`:
```python
app.run(debug=True, port=5003)  # Use a different port
```

## Next Steps

Possible enhancements:
- Add spatial maps (not just time series)
- Compare multiple forecast runs
- Add verification against observations
- Export forecast data to CSV/NetCDF
- Add ensemble forecasts

## References

- [PanguWeather Paper](https://www.nature.com/articles/s41586-023-06185-3)
- [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data)
- [ai-models GitHub](https://github.com/ecmwf-lab/ai-models)
