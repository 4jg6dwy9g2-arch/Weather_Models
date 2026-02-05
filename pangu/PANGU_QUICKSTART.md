# PanguWeather Flask App - Quick Start Guide

## What Was Created

A complete Flask web application for running and visualizing PanguWeather forecasts, styled similarly to your existing Weather Models app.

### Files Created

```
ML_Weather_Models/
├── pangu_app.py                    # Main Flask application
├── start_pangu_app.sh              # Startup script
├── pangu_requirements.txt          # Python dependencies
├── PANGU_README.md                 # Full documentation
├── PANGU_QUICKSTART.md            # This file
└── templates/
    ├── pangu_base.html             # Base template with navbar
    ├── pangu_dashboard.html        # Dashboard for viewing forecasts
    └── pangu_run.html              # Page to run new forecasts
```

## Quick Start (3 Steps)

### 1. Navigate to the Directory

```bash
cd ~/Documents/ML_Weather_Models
```

### 2. Start the App

```bash
./start_pangu_app.sh
```

Or directly with Python:

```bash
python pangu_app.py
```

### 3. Open in Browser

Visit: http://localhost:5002

## Using the App

### Dashboard (Home Page)

- View statistics about your forecast runs
- See the latest forecast chart with:
  - Temperature at 850 hPa (lower atmosphere)
  - Temperature at 500 hPa (mid atmosphere)
  - Wind speed at 850 hPa
- Browse all saved forecast runs
- Click any run to view its data

### Run Forecast Page

1. Click "Run Forecast" in the navigation
2. Configure your forecast:
   - **Date**: Use yesterday (default) since ECMWF data lags 1-2 days
   - **Time**: Choose 00, 06, 12, or 18 UTC
   - **Lead Time**: 24 hours for quick test, up to 168 hours (7 days)
3. Click "Run Forecast"
4. Watch real-time logs as the model executes
5. When complete, view results on dashboard

## Example First Run

```
Date: 2026-01-28 (yesterday)
Time: 12:00 UTC
Lead Time: 24 hours
Expected runtime: ~7-8 minutes on Apple Silicon
```

## Features

✅ **Clean UI**: Modern, responsive design matching your Weather Models app
✅ **Real-time Logs**: Watch model execution via Server-Sent Events
✅ **Interactive Charts**: Zoom, pan, and explore forecast time series
✅ **Run History**: Keep track of up to 50 forecast runs
✅ **Automatic Data Extraction**: GRIB files are processed automatically
✅ **Smart Defaults**: Yesterday's date pre-selected

## What the App Does

1. **Runs PanguWeather** using the ai-models command
2. **Downloads initial conditions** from ECMWF Open Data
3. **Generates forecast** using your pre-downloaded ONNX model files
4. **Processes GRIB output** to extract time series data
5. **Saves everything** to JSON database and GRIB files
6. **Displays results** in interactive charts

## Data Storage

- **GRIB Files**: `pangu_forecasts/` directory (full spatial fields)
- **JSON Database**: `pangu_runs.json` (metadata and time series)
- **Run History**: Last 50 runs kept automatically

## Ports

- **PanguWeather App**: http://localhost:5002
- **Weather Models App**: http://localhost:5001 (if running)

Both apps can run simultaneously!

## Tips

💡 **Start Simple**: Run a 24-hour forecast first to test everything
💡 **Use Yesterday**: ECMWF data typically 1-2 days behind
💡 **Watch Logs**: Real-time log viewer shows progress
💡 **Compare Runs**: Run multiple forecasts with different init times
💡 **Check Dashboard**: Results update automatically after each run

## Troubleshooting

### Model files not found

Make sure you're in the right directory and have the ONNX files:
```bash
cd ~/Documents/ML_Weather_Models
ls -lh *.onnx
```

You should see:
- `pangu_weather_24.onnx` (~1.1 GB)
- `pangu_weather_6.onnx` (~1.1 GB)

### Port already in use

Change port in `pangu_app.py` line 312:
```python
app.run(debug=True, port=5003)  # Use different port
```

### Dependencies missing

```bash
pip install -r pangu_requirements.txt
```

### Can't read GRIB files

```bash
brew install eccodes  # macOS
pip install cfgrib
```

## Next Steps

Once you have a few forecasts running, you might want to:

- Compare forecasts from different init times
- Add spatial maps (currently only time series)
- Verify against observations
- Export data to other formats
- Add more variables to the charts

## Questions?

Check the full documentation in `PANGU_README.md` for more details.

Enjoy exploring AI weather forecasting! 🌦️
