# Claude Code Implementation Log

This document tracks major features and implementations added to the Weather Models project by Claude Code.

---

## Rossby Wave Number Analysis (2026-02-02)

Added functionality to extract 500 hPa geopotential height from weather models (GFS, AIFS, IFS) and calculate the number of Rossby waves in the Northern Hemisphere. Wave numbers are displayed on the dashboard to help assess atmospheric flow patterns and their impact on forecast reliability.

### Implementation Summary

#### 1. Variable Definitions Added
- **GFS** (`app.py`): Added `Z500_GFS` variable with Herbie search string `:HGT:500 mb:`
- **IFS** (`ecmwf_ifs.py`): Added `z500` to `IFS_VARIABLES` with ECMWF param `gh` at level `500`, plus unit conversion for geopotential to geopotential height
- **AIFS**: Already had `z500` defined (no changes needed)

#### 2. Wave Calculation Module Created
- **`rossby_waves.py`**: New module with:
  - `calculate_wave_number()`: Main function that extracts z500 along 55°N, applies smoothing, and uses peak detection to count ridges and troughs
  - FFT analysis for identifying dominant wave numbers
  - Multi-latitude analysis support (for future enhancement)

#### 3. Data Fetching Integration
- Modified three fetch functions in `app.py`:
  - `fetch_gfs_data()`
  - `fetch_aifs_data()`
  - `fetch_ifs_data()`
- Each now fetches global NH z500 field at F000 and calculates wave metrics
- Results stored in `z500_waves` field in returned data

#### 4. Data Persistence
- `save_forecast_data()` already saves complete model data dictionaries, so `z500_waves` is automatically preserved in `forecasts.json`

#### 5. API Endpoints Added
- **`/api/wave-analysis`**: Returns current wave numbers for latest forecast run
- **`/api/wave-time-series`**: Returns historical wave numbers for last 30 days (configurable)

#### 6. Dashboard UI Added
- New "Rossby Wave Analysis" card in `dashboard.html` showing:
  - Current wave numbers for all three models (GFS, AIFS, IFS)
  - Ridge and trough counts
  - Interpretation guide explaining predictability implications
  - Historical time series chart showing wave number trends over time
- JavaScript functions for loading and displaying wave data
- Auto-refresh when location changes

### Key Features
- **Wave Number Calculation**: Uses peak detection along 55°N latitude circle to count ridges and troughs
- **Smoothing**: Applied to remove small-scale noise (5-point running mean)
- **Prominence Filtering**: Only counts peaks/troughs with prominence > 50 dm
- **FFT Validation**: Identifies dominant wave numbers using Fourier analysis
- **Historical Tracking**: Displays trends over time to identify flow regime changes
- **Interpretation**: Lower wave numbers (2-4) = better predictability; higher (5-8+) = reduced predictability

### Technical Details
- Wave calculations only run at F000 (initialization time) to minimize overhead
- Global NH region: 180°W to 180°E, 20°N to 70°N
- Default analysis latitude: 55°N
- Peak prominence threshold: 50 dm (decameters)
- Smoothing window: 5 grid points

### Files Modified
- `app.py` - Added Z500_GFS variable, wave calculation integration, API endpoints
- `ecmwf_ifs.py` - Added z500 variable definition and unit conversion
- `rossby_waves.py` - New file with wave calculation algorithms
- `templates/dashboard.html` - Added wave analysis UI card and JavaScript

### Performance Impact
- Additional data fetch per sync: ~1-2 MB GRIB2 per model
- Wave calculation time: ~10-20 ms per model (negligible)
- Only calculates at F000, not all forecast hours
- Total overhead: ~5-10 seconds per sync for 3 models

### Future Enhancements (Not Yet Implemented)
- Wave-skill correlation analysis (quantify relationship between wave number and forecast MAE)
- Multiple latitude analysis (45°N, 50°N, 55°N, 60°N) with mean/std calculation
- Wave amplitude calculation (ridge/trough depth in decameters)
- Wave pattern classification (blocking vs progressive flow regimes)
- 500 hPa height field map visualization with contours
- Wave speed/phase speed calculation across time

---

## 15-Day Wave Number Forecast (2026-02-02)

Enhanced the Rossby Wave Analysis feature to include a 15-day forecast of wave numbers, allowing users to see how atmospheric flow patterns are expected to evolve over time.

### Implementation Summary

#### 1. Modified Data Fetching Functions
Updated all three model fetch functions to calculate wave numbers at 24-hour intervals:
- **GFS**: F000 to F360 (15 days)
- **AIFS**: F000 to F360 (15 days)
- **IFS**: F000 to F240 (10 days - limited by open data availability)

Each fetch function now returns:
- `z500_waves`: Current wave number at F000 (unchanged)
- `z500_waves_forecast`: Time series containing:
  - `times`: Valid times for each forecast hour
  - `wave_numbers`: Wave number at each forecast hour
  - `ridges`: Ridge count at each forecast hour
  - `troughs`: Trough count at each forecast hour

#### 2. New API Endpoint
- **`/api/wave-forecast`**: Returns 15-day wave number forecast for latest run
  - Includes forecast data from all three models
  - Returns times and wave numbers for plotting

#### 3. Dashboard UI Enhancement
Added new forecast chart to the Rossby Wave Analysis card:
- **15-Day Wave Number Forecast Chart**: Shows predicted evolution of wave numbers
  - All three models displayed on same chart for comparison
  - Interactive tooltips showing wave count and forecast day
  - Automatic loading on page load and location change
- Retained existing "Wave Number History" chart showing past month trends

### Key Features
- **Forecast Evolution**: See how wave patterns are expected to change over 2 weeks
- **Model Comparison**: Compare GFS, AIFS, and IFS wave number forecasts
- **Flow Regime Transitions**: Identify when atmospheric flow may shift between amplified/progressive patterns
- **Predictability Assessment**: Lower, steady wave numbers suggest better extended forecast reliability

### Technical Details
- Wave calculations at 24-hour intervals (F000, F024, F048, ..., F360)
- Each calculation uses same methodology as current analysis (55°N, peak detection, smoothing)
- Forecast data stored in `forecasts.json` alongside other model data
- Chart uses Chart.js with line plot showing all three models

### Performance Impact
- **Increased data fetch**: Now fetches z500 at 16 forecast hours (GFS/AIFS) or 11 (IFS) instead of just F000
- **Additional computation**: ~16x more wave calculations per sync (~160-320 ms per model)
- **Total overhead**: ~30-60 seconds per sync for 3 models (was ~5-10 seconds)
- **Storage**: Minimal (~2-3 KB additional JSON per forecast run)

### Files Modified
- `app.py` - Modified fetch functions, added `/api/wave-forecast` endpoint
- `templates/dashboard.html` - Added forecast chart and JavaScript functions
- `templates/base.html` - Fixed auto-shutdown bug (was triggering on tab changes)

### Bug Fix
Fixed issue where auto-shutdown feature was terminating the Flask server when navigating between tabs. Now only shuts down when truly leaving the site or closing the browser.

---
