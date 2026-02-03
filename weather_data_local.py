"""
Local weather data loading module (self-contained).

- Current conditions: fetched live from WeatherLink API (handled in app.py)
- Historical data: read from CSV files stored in:
  ~/Documents/Townhome_Weather/Weather_Data/YEAR/MMYYYY.csv
- IAD Climatology: 32 years of historical data (1990-2021) from:
  ~/Documents/Townhome_Weather/Weather_Data/Climo/IAD/climo.txt
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np

# Path to weather data
WEATHER_DATA_PATH = Path.home() / "Documents" / "Townhome_Weather" / "Weather_Data"

# Path to IAD climatology data
IAD_CLIMO_PATH = WEATHER_DATA_PATH / "Climo" / "IAD" / "climo.txt"

# Cache for IAD climatology data (loaded once)
_iad_climo_cache = None

# Soil moisture columns that need unit normalization
SOIL_MOISTURE_COLS = [
    "soil_moisture_10in",
    "soil_moisture_5in",
    "soil_moisture_20in",
]

# Column mapping from CSV headers to friendly names
# Includes both old format (with deg F) and new format (with F) for compatibility
# Note: Probe 1 is at 10", Probe 2 is at 5", Probe 4 is at 20"
COLUMN_MAP = {
    "Date & Time": "datetime",
    # Old format (pre-2026)
    "Temp - \u00b0F": "temperature",
    "Dew Point - \u00b0F": "dew_point",
    "Temp 1 - \u00b0F": "soil_temp_10in",
    "Temp 2 - \u00b0F": "soil_temp_5in",
    "Temp 4 - \u00b0F": "soil_temp_20in",
    "Heat Index - \u00b0F": "heat_index",
    # New format (2026+)
    "Temp - F": "temperature",
    "Dew Point - F": "dew_point",
    "Temp 1 - F": "soil_temp_10in",
    "Temp 2 - F": "soil_temp_5in",
    "Temp 4 - F": "soil_temp_20in",
    "Heat Index - F": "heat_index",
    # Common columns (unchanged)
    "Avg Wind Speed - mph": "wind_speed",
    "Prevailing Wind Direction": "wind_dir",
    "High Wind Speed - mph": "wind_gust",
    "Rain - in": "rain",
    "High Rain Rate - in/h": "rain_rate",
    "Solar Rad - W/m^2": "solar_rad",
    "Soil Moisture 1 - bar": "soil_moisture_10in",
    "Soil Moisture 2 - bar": "soil_moisture_5in",
    "Soil Moisture 4 - bar": "soil_moisture_20in",
    "Barometer - mb": "barometer",
}


def _load_iad_climatology() -> Optional[dict]:
    """
    Load IAD climatology data from climo.txt file.

    Returns dict with:
        - "highs": 2D numpy array (32 years x 365 days) of daily high temps
        - "lows": 2D numpy array (32 years x 365 days) of daily low temps
        - "dates": list of (month, day) tuples for each day-of-year
    """
    global _iad_climo_cache
    if _iad_climo_cache is not None:
        return _iad_climo_cache

    if not IAD_CLIMO_PATH.exists():
        return None

    try:
        highs = np.zeros((32, 365))
        lows = np.zeros((32, 365))
        dates = []

        with open(IAD_CLIMO_PATH, "r") as f:
            lines = f.readlines()

        row_idx = 0
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            date_str = parts[0]
            try:
                high = float(parts[1])
                low = float(parts[2])
            except ValueError:
                continue

            date_parts = date_str.split("/")
            if len(date_parts) >= 2:
                month = int(date_parts[0])
                day = int(date_parts[1])
            else:
                continue

            year_in_set = row_idx % 32
            day_of_year = row_idx // 32

            if day_of_year < 365:
                highs[year_in_set, day_of_year] = high
                lows[year_in_set, day_of_year] = low

                if year_in_set == 0 and day_of_year == len(dates):
                    dates.append((month, day))

            row_idx += 1

        _iad_climo_cache = {
            "highs": highs,
            "lows": lows,
            "dates": dates,
        }
        return _iad_climo_cache

    except Exception:
        return None


def _get_day_of_year_index(month: int, day: int) -> Optional[int]:
    """Get the day-of-year index (0-364) for a given month/day."""
    climo = _load_iad_climatology()
    if climo is None:
        return None

    try:
        return climo["dates"].index((month, day))
    except ValueError:
        if month == 2 and day == 29:
            try:
                return climo["dates"].index((2, 28))
            except ValueError:
                return None
        return None


def get_climatology_for_date(date) -> Optional[dict]:
    """
    Get IAD climatology statistics for a specific date.

    Returns dict with percentiles and median for high and low temps:
        - high_median, high_p25, high_p75, high_p95
        - low_median, low_p05, low_p25, low_p75
    """
    climo = _load_iad_climatology()
    if climo is None:
        return None

    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d").date()
    elif isinstance(date, datetime):
        date = date.date()

    day_idx = _get_day_of_year_index(date.month, date.day)
    if day_idx is None:
        return None

    highs = climo["highs"][:, day_idx]
    lows = climo["lows"][:, day_idx]

    return {
        "high_median": round(float(np.median(highs)), 1),
        "high_p25": round(float(np.percentile(highs, 25)), 1),
        "high_p75": round(float(np.percentile(highs, 75)), 1),
        "high_p95": round(float(np.percentile(highs, 95)), 1),
        "low_median": round(float(np.median(lows)), 1),
        "low_p05": round(float(np.percentile(lows, 5)), 1),
        "low_p25": round(float(np.percentile(lows, 25)), 1),
        "low_p75": round(float(np.percentile(lows, 75)), 1),
    }


def _normalize_soil_moisture(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize soil moisture values to centibars (cb).

    Old data: values like 0.15 (bars) -> convert to 15 (cb)
    New data: values like 15 (cb) -> keep as is
    """
    for col in SOIL_MOISTURE_COLS:
        if col in df.columns:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                median_val = valid_values.median()
                if median_val < 1:
                    df[col] = df[col] * 100
    return df


def _get_csv_files_for_range(start_date: datetime, end_date: datetime) -> list[Path]:
    files = []
    current = datetime(start_date.year, start_date.month, 1)

    while current <= end_date:
        year_dir = WEATHER_DATA_PATH / str(current.year)
        filename = f"{current.month:02d}{current.year}.csv"
        filepath = year_dir / filename

        if filepath.exists():
            files.append(filepath)

        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    return files


def _load_csv(filepath: Path) -> Optional[pd.DataFrame]:
    if not filepath.exists():
        return None

    try:
        df = pd.read_csv(
            filepath,
            skiprows=5,
            encoding="latin1",
            na_values=["--", ""],
        )
    except Exception:
        return None

    df = df.rename(columns=COLUMN_MAP)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], format="%m/%d/%y %H:%M")
    else:
        return None

    df = _normalize_soil_moisture(df)

    return df


def get_historical_data(start_date: datetime, end_date: datetime) -> list[dict]:
    files = _get_csv_files_for_range(start_date, end_date)
    if not files:
        return []

    dfs = []
    for f in files:
        df = _load_csv(f)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        return []

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[
        (combined["datetime"] >= start_date) &
        (combined["datetime"] <= end_date)
    ]

    combined = combined.sort_values("datetime")

    records = []
    for _, row in combined.iterrows():
        def safe_value(val, decimals=1):
            if pd.isna(val):
                return None
            if isinstance(val, float):
                return round(val, decimals)
            return val

        records.append({
            "datetime": row["datetime"].isoformat() if pd.notna(row.get("datetime")) else None,
            "temperature": safe_value(row.get("temperature")),
            "dew_point": safe_value(row.get("dew_point")),
            "wind_speed": safe_value(row.get("wind_speed")),
            "rain": safe_value(row.get("rain"), 2),
            "solar_rad": safe_value(row.get("solar_rad"), 0),
            "soil_temp_5in": safe_value(row.get("soil_temp_5in")),
            "soil_temp_10in": safe_value(row.get("soil_temp_10in")),
            "soil_temp_20in": safe_value(row.get("soil_temp_20in")),
            "soil_moisture_5in": safe_value(row.get("soil_moisture_5in")),
            "soil_moisture_10in": safe_value(row.get("soil_moisture_10in")),
            "soil_moisture_20in": safe_value(row.get("soil_moisture_20in")),
        })

    return records


def get_daily_summaries(start_date: datetime, end_date: datetime) -> list[dict]:
    files = _get_csv_files_for_range(start_date, end_date)
    if not files:
        return []

    dfs = []
    for f in files:
        df = _load_csv(f)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        return []

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[
        (combined["datetime"] >= start_date) &
        (combined["datetime"] <= end_date)
    ]

    if combined.empty:
        return []

    combined["date"] = combined["datetime"].dt.date

    summaries = []
    for date, group in combined.groupby("date"):
        def safe_agg(series, func):
            try:
                val = func(series.dropna())
                return round(val, 1) if not pd.isna(val) else None
            except (ValueError, TypeError):
                return None

        rain_values = group["rain"].dropna() if "rain" in group else []
        daily_rain = round(rain_values.max(), 2) if len(rain_values) > 0 else 0

        daylight_solar = group["solar_rad"].dropna() if "solar_rad" in group else []
        daylight_solar = daylight_solar[daylight_solar > 0] if len(daylight_solar) > 0 else []

        temp_series = group["temperature"] if "temperature" in group else pd.Series(dtype=float)
        dew_series = group["dew_point"] if "dew_point" in group else pd.Series(dtype=float)
        wind_series = group["wind_speed"] if "wind_speed" in group else pd.Series(dtype=float)
        gust_series = group["wind_gust"] if "wind_gust" in group else pd.Series(dtype=float)
        soil5 = group["soil_temp_5in"] if "soil_temp_5in" in group else pd.Series(dtype=float)
        soil10 = group["soil_temp_10in"] if "soil_temp_10in" in group else pd.Series(dtype=float)
        soil20 = group["soil_temp_20in"] if "soil_temp_20in" in group else pd.Series(dtype=float)
        moist5 = group["soil_moisture_5in"] if "soil_moisture_5in" in group else pd.Series(dtype=float)
        moist10 = group["soil_moisture_10in"] if "soil_moisture_10in" in group else pd.Series(dtype=float)
        moist20 = group["soil_moisture_20in"] if "soil_moisture_20in" in group else pd.Series(dtype=float)

        summaries.append({
            "date": date.isoformat(),
            "temp_high": safe_agg(temp_series, np.max),
            "temp_low": safe_agg(temp_series, np.min),
            "temp_avg": safe_agg(temp_series, np.mean),
            "dew_point_avg": safe_agg(dew_series, np.mean),
            "wind_speed_avg": safe_agg(wind_series, np.mean),
            "wind_gust_max": safe_agg(gust_series, np.max),
            "rain_total": daily_rain,
            "solar_rad_avg": safe_agg(daylight_solar, np.mean) if len(daylight_solar) > 0 else None,
            "solar_rad_max": safe_agg(daylight_solar, np.max) if len(daylight_solar) > 0 else None,
            "soil_temp_5in_avg": safe_agg(soil5, np.mean),
            "soil_temp_10in_avg": safe_agg(soil10, np.mean),
            "soil_temp_20in_avg": safe_agg(soil20, np.mean),
            "soil_moisture_5in_avg": safe_agg(moist5, np.mean),
            "soil_moisture_10in_avg": safe_agg(moist10, np.mean),
            "soil_moisture_20in_avg": safe_agg(moist20, np.mean),
        })

    return summaries


def get_period_summary(start_date: datetime, end_date: datetime) -> dict:
    daily = get_daily_summaries(start_date, end_date)
    if not daily:
        return {}

    df = pd.DataFrame(daily)

    def safe_stat(series, func):
        try:
            val = func(series.dropna())
            return round(val, 1) if not pd.isna(val) else None
        except (ValueError, TypeError):
            return None

    solar_rad_avg = safe_stat(df["solar_rad_avg"], lambda x: x.mean()) if "solar_rad_avg" in df else None
    solar_rad_historical = _get_historical_solar_rad_avg(start_date, end_date)

    wind_speed_avg = safe_stat(df["wind_speed_avg"], lambda x: x.mean()) if "wind_speed_avg" in df else None
    wind_gust_avg = safe_stat(df["wind_gust_max"], lambda x: x.mean()) if "wind_gust_max" in df else None
    wind_historical = _get_historical_wind_avg(start_date, end_date)

    return {
        "days": len(daily),
        "temp_max": safe_stat(df["temp_high"], max) if "temp_high" in df else None,
        "temp_min": safe_stat(df["temp_low"], min) if "temp_low" in df else None,
        "temp_avg": safe_stat(df["temp_avg"], lambda x: x.mean()) if "temp_avg" in df else None,
        "rain_total": round(df["rain_total"].sum(), 2) if "rain_total" in df else 0,
        "rainy_days": int((df["rain_total"] > 0).sum()) if "rain_total" in df else 0,
        "solar_rad_avg": solar_rad_avg,
        "solar_rad_historical": solar_rad_historical,
        "wind_speed_avg": wind_speed_avg,
        "wind_gust_avg": wind_gust_avg,
        "wind_speed_historical": wind_historical.get("speed") if wind_historical else None,
        "wind_gust_historical": wind_historical.get("gust") if wind_historical else None,
    }


def _get_historical_solar_rad_avg(start_date: datetime, end_date: datetime) -> Optional[float]:
    year_offset = end_date.year - start_date.year
    historical_years = list(range(2020, start_date.year))
    if not historical_years:
        return None

    all_solar_avgs = []
    for year in historical_years:
        try:
            hist_start = start_date.replace(year=year)
            hist_end = end_date.replace(year=year + year_offset)
        except ValueError:
            continue

        daily = get_daily_summaries(hist_start, hist_end)
        if daily:
            solar_vals = [d["solar_rad_avg"] for d in daily if d.get("solar_rad_avg") is not None]
            if solar_vals:
                all_solar_avgs.extend(solar_vals)

    if all_solar_avgs:
        return round(sum(all_solar_avgs) / len(all_solar_avgs), 1)
    return None


def _get_historical_wind_avg(start_date: datetime, end_date: datetime) -> Optional[dict]:
    year_offset = end_date.year - start_date.year
    historical_years = list(range(2020, start_date.year))
    if not historical_years:
        return None

    all_speed_avgs = []
    all_gust_avgs = []

    for year in historical_years:
        try:
            hist_start = start_date.replace(year=year)
            hist_end = end_date.replace(year=year + year_offset)
        except ValueError:
            continue

        daily = get_daily_summaries(hist_start, hist_end)
        if daily:
            speed_vals = [d["wind_speed_avg"] for d in daily if d.get("wind_speed_avg") is not None]
            gust_vals = [d["wind_gust_max"] for d in daily if d.get("wind_gust_max") is not None]
            if speed_vals:
                all_speed_avgs.extend(speed_vals)
            if gust_vals:
                all_gust_avgs.extend(gust_vals)

    result = {}
    if all_speed_avgs:
        result["speed"] = round(sum(all_speed_avgs) / len(all_speed_avgs), 1)
    if all_gust_avgs:
        result["gust"] = round(sum(all_gust_avgs) / len(all_gust_avgs), 1)

    return result if result else None


def get_daily_summaries_with_climo(start_date: datetime, end_date: datetime) -> dict:
    daily = get_daily_summaries(start_date, end_date)
    climo = []
    anomalies = []

    for d in daily:
        climo_stats = get_climatology_for_date(d["date"]) or {}
        climo.append(climo_stats)

        anomaly = None
        running_mean = None
        if d.get("temp_high") is not None and d.get("temp_low") is not None and climo_stats.get("high_median") is not None and climo_stats.get("low_median") is not None:
            high_anomaly = d["temp_high"] - climo_stats["high_median"]
            low_anomaly = d["temp_low"] - climo_stats["low_median"]
            anomaly = round((high_anomaly + low_anomaly) / 2 - 1.0, 1)
        anomalies.append({
            "anomaly": anomaly,
            "running_mean": running_mean,
        })

    # Compute running mean for anomalies
    window = 7
    for i in range(len(anomalies)):
        window_vals = [a["anomaly"] for a in anomalies[max(0, i - window + 1):i + 1] if a["anomaly"] is not None]
        if window_vals:
            anomalies[i]["running_mean"] = round(sum(window_vals) / len(window_vals), 2)

    return {
        "daily": daily,
        "climo": climo,
        "anomalies": anomalies,
    }


def get_soil_moisture_percentiles(current_values: dict) -> dict:
    """
    Calculate percentiles for current soil moisture readings based on all historical data.

    Args:
        current_values: dict with keys 'soil_moisture_5in', 'soil_moisture_10in', 'soil_moisture_20in'

    Returns:
        dict with percentile for each depth (0-100), or None if insufficient data
    """
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()

    files = _get_csv_files_for_range(start_date, end_date)

    if not files:
        return {
            "soil_moisture_5in_percentile": None,
            "soil_moisture_10in_percentile": None,
            "soil_moisture_20in_percentile": None,
        }

    dfs = []
    for f in files:
        df = _load_csv(f)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        return {
            "soil_moisture_5in_percentile": None,
            "soil_moisture_10in_percentile": None,
            "soil_moisture_20in_percentile": None,
        }

    combined = pd.concat(dfs, ignore_index=True)

    result = {}
    for depth in ["5in", "10in", "20in"]:
        col = f"soil_moisture_{depth}"
        current_val = current_values.get(col)

        if current_val is None or col not in combined.columns:
            result[f"{col}_percentile"] = None
            continue

        historical = combined[col].dropna()

        if len(historical) < 10:
            result[f"{col}_percentile"] = None
            continue

        percentile = (historical <= current_val).sum() / len(historical) * 100
        result[f"{col}_percentile"] = round(percentile)

    return result
