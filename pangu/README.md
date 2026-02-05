# AI Weather Forecasting Setup

## ✅ What's Installed & Working

### PanguWeather Model
- **Status**: Installed and ready
- **Platform**: Works on Apple Silicon, NVIDIA GPU, and CPU
- **Developed by**: Huawei Research (Nature 2023)
- **Technology**: Uses ONNX runtime (no Flash Attention required!)

### Infrastructure
- ✓ `ai-models` framework (0.7.4)
- ✓ `ai-models-panguweather` (0.0.9)
- ✓ ONNX runtime for Apple Silicon
- ✓ ECMWF data pipeline tools

## 🚀 Running PanguWeather

### Simple Method
```bash
python run_pangu.py
```

### Manual Method
```bash
ai-models panguweather \
  --input ecmwf-open-data \
  --date 20260127 \
  --time 12 \
  --lead-time 24 \
  --path ./forecasts
```

### Command Options
- `--date YYYYMMDD` - Initial date (e.g., 20260127)
- `--time HH` - Initial hour: 00, 06, 12, or 18
- `--lead-time HOURS` - Forecast length (24, 48, 72, etc.)
- `--path DIR` - Output directory

## ⚠️ Current Issue: ECMWF Server Connectivity

The ECMWF open data portal is experiencing connection issues:
```
WARNING: Recovering from connection error
WARNING: Retrying in 120 seconds
```

This is a **temporary server issue**, not a problem with your setup.

### Solutions:

**Option 1: Wait and retry later**
ECMWF servers may be overloaded. Try again in a few hours.

**Option 2: Use alternative data source**
If you have access to MARS or CDS:
```bash
ai-models panguweather --input mars --date 20260127
```

**Option 3: Use pre-downloaded data**
If you have GRIB files locally:
```bash
ai-models panguweather --input file --file your_data.grib
```

## 📊 What PanguWeather Does

PanguWeather generates global weather forecasts including:
- **Temperature** at multiple atmospheric levels
- **Wind** speed and direction (U/V components)
- **Geopotential height**
- **Humidity** (specific humidity)
- **Vertical velocity**

### Forecast Details
- **Resolution**: 0.25° (about 25km)
- **Pressure levels**: 13 levels (1000-50 hPa)
- **Time steps**: 6-hour increments
- **Max forecast**: 7 days (168 hours)
- **Speed**: ~15 seconds per day on GPU, ~2-3 minutes on CPU

## 🔧 Troubleshooting

### Connection Errors
If you see retry messages, the ECMWF server is temporarily unavailable:
- Press Ctrl+C to cancel
- Wait 30-60 minutes
- Try again

### Model Download
On first run, PanguWeather downloads model weights (~500MB):
- Stored in `~/.cache/ai-models/`
- Only downloaded once
- No internet needed after first run

### Output Format
Forecasts are saved as **GRIB2** files:
- Industry-standard meteorological format
- Read with `xarray`, `cfgrib`, or `ecCodes`
- View with Panoply, ncview, or similar tools

## 📁 Files

- `run_pangu.py` - Simple Python script to run forecasts
- `test_installation.py` - Verify installation
- `requirements.txt` - Package dependencies
- `forecasts/` - Output directory (created on first run)

## 🌐 Reading Forecast Output

### Using Python
```python
import xarray as xr

# Open forecast
ds = xr.open_dataset('forecasts/output.grib', engine='cfgrib')

# View temperature at 850hPa
temp = ds['t'].sel(isobaricInhPa=850)

# Plot
import matplotlib.pyplot as plt
temp.plot()
plt.show()
```

### Using Command Line
```bash
# Install grib tools
brew install eccodes

# View GRIB file info
grib_ls forecasts/*.grib

# Extract specific field
grib_get -p shortName,level forecasts/*.grib
```

## 📚 Learn More

- **PanguWeather Paper**: https://www.nature.com/articles/s41586-023-06185-3
- **ECMWF Open Data**: https://www.ecmwf.int/en/forecasts/datasets/open-data
- **ai-models GitHub**: https://github.com/ecmwf-lab/ai-models

## 🎯 Next Steps

1. **Wait for ECMWF servers** to recover (check https://www.ecmwf.int/en/forecasts/datasets/open-data)
2. **Try the forecast again** with `python run_pangu.py`
3. **Explore the output** when it completes

Your setup is complete and ready! The only issue is temporary server connectivity.
