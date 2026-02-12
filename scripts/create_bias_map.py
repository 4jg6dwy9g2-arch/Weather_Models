#!/usr/bin/env python3
"""
Create a bias map for ASOS stations for Feb 7 2026 12Z
Shows temperature bias (forecast - observation) for GFS, AIFS, and IFS
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import folium
from folium import plugins
import numpy as np

# Target valid time: Feb 7 12Z
target_time = datetime(2026, 2, 7, 12, 0, 0)

# Use Feb 7 00Z run at F+12 hours
run_time = datetime(2026, 2, 7, 0, 0, 0)
lead_hours = 12

print(f"Creating bias map for valid time: {target_time.strftime('%Y-%m-%d %HZ')}")
print(f"Using run: {run_time.strftime('%Y-%m-%d %HZ')} + {lead_hours}h")

# Load verification cache
print("\nLoading ASOS verification cache...")
with open(Path(__file__).parent.parent / 'data' / 'asos_verification_cache.json', 'r') as f:
    cache = json.load(f)

# Load station metadata
stations_meta = cache['stations']
by_station = cache['by_station']

print(f"Total stations in cache: {len(stations_meta)}")

# Extract bias data for each station
bias_data = {
    'gfs': [],
    'aifs': [],
    'ifs': []
}

station_count = 0
for station_id, station_info in stations_meta.items():
    if station_id not in by_station:
        continue

    station_data = by_station[station_id]
    lat = station_info['lat']
    lon = station_info['lon']
    name = station_info['name']

    # Get forecast and observation for each model at F+12
    for model in ['gfs', 'aifs', 'ifs']:
        if model not in station_data:
            continue

        model_data = station_data[model]

        # Check if we have data for this lead time
        lead_key = str(lead_hours)
        if lead_key not in model_data:
            continue

        lead_data = model_data[lead_key]

        # Check for temperature bias
        if 'temp' in lead_data:
            temp_stats = lead_data['temp']
            if isinstance(temp_stats, dict) and 'bias' in temp_stats:
                bias = temp_stats['bias']
                count = temp_stats.get('count', 0)

                if count > 0:
                    bias_data[model].append({
                        'station_id': station_id,
                        'name': name,
                        'lat': lat,
                        'lon': lon,
                        'bias': bias,
                        'count': count
                    })

    station_count += 1

# Print summary
print(f"\nStations processed: {station_count}")
for model in ['gfs', 'aifs', 'ifs']:
    print(f"{model.upper()}: {len(bias_data[model])} stations with bias data")

# Create map
print("\nCreating interactive map...")

# Center map on CONUS
m = folium.Map(
    location=[39.8, -98.5],
    zoom_start=4,
    tiles='CartoDB positron'
)

# Color scale: blue (cold bias) to red (warm bias)
def get_color(bias):
    """Return color based on bias magnitude"""
    if bias < -5:
        return '#0000FF'  # Dark blue (strong cold bias)
    elif bias < -2:
        return '#4169E1'  # Royal blue
    elif bias < -1:
        return '#87CEEB'  # Sky blue
    elif bias < 1:
        return '#90EE90'  # Light green (near neutral)
    elif bias < 2:
        return '#FFD700'  # Gold
    elif bias < 5:
        return '#FF8C00'  # Dark orange
    else:
        return '#FF0000'  # Red (strong warm bias)

def get_radius(bias):
    """Return marker size based on bias magnitude"""
    return min(max(abs(bias) * 0.5 + 3, 3), 10)

# Create feature groups for each model
fg_gfs = folium.FeatureGroup(name='GFS', show=True)
fg_aifs = folium.FeatureGroup(name='AIFS', show=False)
fg_ifs = folium.FeatureGroup(name='IFS', show=False)

feature_groups = {
    'gfs': fg_gfs,
    'aifs': fg_aifs,
    'ifs': fg_ifs
}

# Add stations to map
for model, fg in feature_groups.items():
    for station in bias_data[model]:
        color = get_color(station['bias'])
        radius = get_radius(station['bias'])

        # Create popup with station info
        popup_html = f"""
        <b>{station['station_id']}</b> - {station['name']}<br>
        Model: {model.upper()}<br>
        Bias: {station['bias']:.1f}°F<br>
        Verifications: {station['count']}<br>
        Lat: {station['lat']:.4f}, Lon: {station['lon']:.4f}
        """

        folium.CircleMarker(
            location=[station['lat'], station['lon']],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=300),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=1
        ).add_to(fg)

    fg.add_to(m)

# Add layer control
folium.LayerControl(collapsed=False).add_to(m)

# Add legend
legend_html = """
<div style="position: fixed;
     bottom: 50px; right: 50px; width: 200px; height: 300px;
     background-color: white; border:2px solid grey; z-index:9999;
     font-size:14px; padding: 10px">

<h4 style="margin-top:0">Temperature Bias</h4>
<p style="margin: 5px 0; font-size: 12px">Forecast - Observed</p>

<div style="margin-top: 10px">
<i style="background:#FF0000; width: 20px; height: 20px; float: left; margin-right: 8px"></i>
> +5°F (warm bias)<br>

<i style="background:#FF8C00; width: 20px; height: 20px; float: left; margin-right: 8px"></i>
+2 to +5°F<br>

<i style="background:#FFD700; width: 20px; height: 20px; float: left; margin-right: 8px"></i>
+1 to +2°F<br>

<i style="background:#90EE90; width: 20px; height: 20px; float: left; margin-right: 8px"></i>
-1 to +1°F<br>

<i style="background:#87CEEB; width: 20px; height: 20px; float: left; margin-right: 8px"></i>
-2 to -1°F<br>

<i style="background:#4169E1; width: 20px; height: 20px; float: left; margin-right: 8px"></i>
-5 to -2°F<br>

<i style="background:#0000FF; width: 20px; height: 20px; float: left; margin-right: 8px"></i>
< -5°F (cold bias)<br>
</div>

<p style="margin-top: 10px; font-size: 11px; color: #666">
Valid: Feb 7 2026 12Z<br>
Run: Feb 7 00Z + 12h
</p>
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))

# Add title
title_html = """
<div style="position: fixed;
     top: 10px; left: 50px; width: 400px; height: 60px;
     background-color: white; border:2px solid grey; z-index:9999;
     font-size:16px; padding: 10px">
<h3 style="margin:0">ASOS Temperature Bias Map</h3>
<p style="margin: 5px 0; font-size: 12px">Valid Time: February 7, 2026 12Z</p>
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

# Save map
output_file = 'asos_bias_map_feb7_12z.html'
m.save(output_file)

print(f"\nMap saved to: {output_file}")
print("\nSummary Statistics:")
print("-" * 60)

for model in ['gfs', 'aifs', 'ifs']:
    if bias_data[model]:
        biases = [s['bias'] for s in bias_data[model]]
        print(f"\n{model.upper()}:")
        print(f"  Stations: {len(biases)}")
        print(f"  Mean bias: {np.mean(biases):.2f}°F")
        print(f"  Median bias: {np.median(biases):.2f}°F")
        print(f"  Std dev: {np.std(biases):.2f}°F")
        print(f"  Min bias: {np.min(biases):.2f}°F")
        print(f"  Max bias: {np.max(biases):.2f}°F")
        print(f"  Abs mean bias: {np.mean(np.abs(biases)):.2f}°F")

print("\n" + "=" * 60)
print("Open the HTML file in your browser to view the interactive map")
print("Use the layer control to switch between models")
