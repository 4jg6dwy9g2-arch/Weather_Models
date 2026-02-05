# PanguWeather Map Viewer Guide

## New Features Added! 🗺️

I've added a complete spatial visualization system to view global forecast data with variable selection and regional zooming.

## What's New

### 1. Interactive Map Viewer
- **Full spatial visualization** of forecast fields
- **Variable selection**: Choose from temperature, wind, humidity, geopotential, pressure
- **Level selection**: View any pressure level (1000-50 hPa) or surface
- **Regional views**: Zoom into specific regions
- **Time stepping**: Animate through forecast time steps
- **Statistics**: Min/max/mean/std for each field
- **Download**: Save maps as PNG images

### 2. Regions Available
- **Global** - Full world view
- **North America** - Continental US, Canada, Mexico
- **South America** - Full continent
- **Europe** - Western to Eastern Europe
- **Asia** - Indian subcontinent to East Asia
- **Africa** - Full continent
- **Australia** - Australia and New Zealand
- **Pacific** - Pacific Ocean view
- **Atlantic** - Atlantic Ocean view

## How to Use

### Starting the App

If the app isn't running:
```bash
cd ~/Documents/ML_Weather_Models
python pangu_app.py
```

Visit: http://localhost:5002

### Using the Map Viewer

1. **Click "Map Viewer"** in the navigation menu
2. **Select a forecast run** from the dropdown
3. **Choose your variable**:
   - Temperature (t)
   - U-Wind (u) or V-Wind (v)
   - Geopotential height (z)
   - Humidity (q)
   - Sea level pressure (msl)
4. **Select pressure level** (850 hPa is good for weather, 500 hPa for mid-atmosphere)
5. **Pick a region** (start with Global, then zoom to North America)
6. **Adjust time step** using the slider
7. **Click "Update Map"**

The map will be generated showing your selected field with coastlines, borders, and a color scale.

## Variables Explained

### Temperature (t)
- Shows temperature in Kelvin
- 850 hPa (~5,000 ft) - good for surface weather
- 500 hPa (~18,000 ft) - mid-atmosphere
- Colormap: Red (warm) to Blue (cold)

### Wind Components (u, v)
- u: East-west wind (positive = eastward)
- v: North-south wind (positive = northward)
- 850 hPa shows low-level jets
- Colormap: Blue (westward/southward) to Red (eastward/northward)

### Geopotential Height (z)
- Height of pressure surfaces
- 500 hPa is standard for tracking weather systems
- Ridges (high values) = high pressure
- Troughs (low values) = low pressure
- Colormap: Purple (low) to Yellow (high)

### Specific Humidity (q)
- Water vapor content in kg/kg
- Higher values = more moisture
- 850 hPa good for precipitation forecasts
- Colormap: Yellow (dry) to Blue (moist)

### Sea Level Pressure (msl)
- Surface pressure in Pascals
- Shows high/low pressure systems
- Colormap: Blue (low) to Red (high)

## Tips

💡 **Start with Temperature at 850 hPa** - easiest to interpret
💡 **Use Global view first** - then zoom to your region
💡 **Try different times** - watch weather systems evolve
💡 **Compare levels** - see how patterns change with altitude
💡 **Download maps** - save interesting patterns

## Example Workflows

### Viewing a Cold Front
1. Variable: Temperature (t)
2. Level: 850 hPa
3. Region: North America
4. Step through time to watch front move

### Upper-Level Pattern
1. Variable: Geopotential (z)
2. Level: 500 hPa
3. Region: Global or North America
4. Look for ridges and troughs

### Moisture Transport
1. Variable: Humidity (q)
2. Level: 850 hPa
3. Region: Choose your area
4. Watch moisture plumes

### Low Pressure Systems
1. Variable: Sea Level Pressure (msl)
2. Level: Surface
3. Region: Global or regional
4. Track cyclones and anticyclones

## Technical Details

### Map Generation
- Uses **Cartopy** for projections and features
- **Matplotlib** for plotting
- **15 contour levels** for smooth visualization
- **PlateCarree projection** (lat/lon grid)
- Coastlines, borders, land/ocean features included

### Performance
- Map generation takes 2-5 seconds per image
- Images are ~200-400 KB PNG files
- No caching yet - regenerated each time
- Regional views are faster than global

### Data Resolution
- Original: 0.25° (~25 km)
- Maps show full resolution
- Statistics computed on full grid

## Troubleshooting

### "Variable not found"
Some variables only exist at certain levels:
- msl: surface only
- t2m: surface only
- t, u, v, z, q: pressure levels only

### Blank or strange maps
- Check your time index is valid (0-4 for 24hr forecast)
- Ensure the GRIB file exists
- Try a different variable/level combination

### Slow map generation
- Normal for global views (2-5 seconds)
- Regional views are faster
- First load downloads data from GRIB file

## Next Steps

Possible future enhancements:
- Add wind vectors/streamlines
- Overlay multiple variables
- Animation export (GIF/MP4)
- Custom color scales
- Station data overlay
- Comparison with observations

## Keyboard Shortcuts

Coming soon! For now use the UI controls.

## Questions?

The map viewer uses the same GRIB files as your existing forecasts - no additional data needed!

Enjoy exploring your PanguWeather forecasts visually! 🌦️🗺️
