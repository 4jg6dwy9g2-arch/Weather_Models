#!/usr/bin/env python3
"""
Generate a static GitHub Pages site for ASOS verification.

Writes index.html + data/*.json to the gh-pages worktree, then
commits and pushes to origin/gh-pages.

Usage:
    python export_verification_site.py           # generate only
    python export_verification_site.py --push    # generate + push
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import asos
import export_verification_table as evt

PAGES_WORKTREE = Path('/Users/kennypratt/weather-pages')

# Lead times to publish (matches the map dropdown)
PUBLISH_LEAD_TIMES = [6, 12, 24, 48, 72, 120, 168, 240, 288, 360]
MODELS = ['gfs', 'aifs', 'ifs', 'nws']
VARIABLES = ['temp', 'precip', 'dewpoint', 'mslp']
TS_LEAD_TIMES = [6, 24, 72, 168]   # pre-bake these for the time series chart
SPOTLIGHT_STATIONS = ['IAD', 'BWI']
SPOTLIGHT_LEAD_TIMES = [24, 72, 168]


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def extract_station_data() -> dict:
    """
    Extract per-station MAE/bias from the verification cache in a compact format.

    Returns a dict with:
      lead_times, models, vars, stations
    where stations[sid].d[model_idx][var_idx][lt_idx] = [mae, bias] or null.
    """
    cache = asos.load_verification_cache()
    if cache is None:
        raise RuntimeError("Verification cache not available — run a sync first")

    cache_lead_times = set(cache.get('lead_times', []))
    by_station = cache.get('by_station', {})
    stations_meta = cache.get('stations', {})

    available_lts = [lt for lt in PUBLISH_LEAD_TIMES if lt in cache_lead_times]

    result = {
        'lead_times': available_lts,
        'models': MODELS,
        'vars': VARIABLES,
        'stations': {}
    }

    for station_id, st_cache_data in by_station.items():
        meta = stations_meta.get(station_id, {})
        lat = meta.get('lat')
        lon = meta.get('lon')
        if lat is None or lon is None:
            continue

        # d[model_idx][var_idx][lt_idx] = [mae, bias] or null
        model_arr = []
        has_any = False
        for model in MODELS:
            m_data = st_cache_data.get(model, {})
            var_arr = []
            for var in VARIABLES:
                lt_arr = []
                for lt in available_lts:
                    if model == 'nws' and lt > 168:
                        lt_arr.append(None)
                        continue
                    entry = m_data.get(str(lt), {})
                    v = entry.get(var) if entry else None
                    if v and v.get('count', 0) > 0:
                        lt_arr.append([v['mae'], v['bias']])
                        has_any = True
                    else:
                        lt_arr.append(None)
                var_arr.append(lt_arr)
            model_arr.append(var_arr)

        if not has_any:
            continue

        result['stations'][station_id] = {
            'name': meta.get('name', station_id),
            'lat': round(lat, 4),
            'lon': round(lon, 4),
            'd': model_arr
        }

    return result


def extract_monthly_station_data(available_lts: list) -> dict:
    """
    Extract per-station MAE/bias from the monthly stats cache (last 20 days).

    Uses the same compact d[model_idx][var_idx][lt_idx] = [mae, bias] format
    so the frontend can switch between all-time and monthly with one flag.
    Requires the all-time cache for lat/lon metadata.
    """
    monthly = asos.load_monthly_stats_cache().get('by_station_monthly', {})
    # Get lat/lon from the verification cache (already loaded by caller)
    cache = asos.load_verification_cache()
    stations_meta = cache.get('stations', {}) if cache else {}

    result = {}
    for station_id, st_monthly in monthly.items():
        meta = stations_meta.get(station_id, {})
        lat = meta.get('lat')
        lon = meta.get('lon')
        if lat is None or lon is None:
            continue

        model_arr = []
        has_any = False
        for model in MODELS:
            m_data = st_monthly.get(model, {})
            var_arr = []
            for var in VARIABLES:
                lt_arr = []
                var_data = m_data.get(var, {})
                for lt in available_lts:
                    if model == 'nws' and lt > 168:
                        lt_arr.append(None)
                        continue
                    lt_stats = var_data.get(str(lt))
                    if lt_stats and lt_stats.get('count', 0) > 0:
                        count = lt_stats['count']
                        mae = round(lt_stats['sum_abs_errors'] / count, 2)
                        bias = round(lt_stats['sum_errors'] / count, 2)
                        lt_arr.append([mae, bias])
                        has_any = True
                    else:
                        lt_arr.append(None)
                var_arr.append(lt_arr)
            model_arr.append(var_arr)

        if not has_any:
            continue

        result[station_id] = {
            'name': meta.get('name', station_id),
            'lat': round(lat, 4),
            'lon': round(lon, 4),
            'd': model_arr
        }

    return result


def extract_timeseries_data() -> dict:
    """Pre-bake ASOS time series for all variables × TS_LEAD_TIMES."""
    ts = {}
    for var in VARIABLES:
        ts[var] = {}
        for lt in TS_LEAD_TIMES:
            try:
                gfs = asos.get_verification_time_series_from_cache('gfs', var, lt, days_back=30)
                aifs_d = asos.get_verification_time_series_from_cache('aifs', var, lt, days_back=30)
                ifs_d = asos.get_verification_time_series_from_cache('ifs', var, lt, days_back=30)
                include_nws = var in ('temp', 'precip', 'dewpoint')
                nws_d = asos.get_verification_time_series_from_cache('nws', var, lt, days_back=30) if include_nws else {}

                if 'error' in gfs:
                    continue

                combo = {
                    'dates': gfs.get('dates', []),
                    'gfs':  {'mae': gfs.get('mae', []),   'bias': gfs.get('bias', [])},
                    'aifs': {'mae': aifs_d.get('mae', []), 'bias': aifs_d.get('bias', [])},
                    'ifs':  {'mae': ifs_d.get('mae', []),  'bias': ifs_d.get('bias', [])},
                }
                if include_nws and 'error' not in nws_d:
                    combo['nws'] = {'mae': nws_d.get('mae', []), 'bias': nws_d.get('bias', [])}

                ts[var][str(lt)] = combo
            except Exception as e:
                print(f"  Warning: time series {var}/{lt}h failed: {e}")
    return ts


def extract_spotlight_obs(station_ids: list, lead_times: list) -> dict:
    """
    Extract per-timestamp obs + model biases for spotlight stations at key lead times.

    Returns {station_id: {lt_str: [{t, ot, op, od, tb, pb, db}, ...]}}
    t=time ISO, ot/op/od=obs temp/precip/dewpoint, tb/pb/db=bias dicts {model: float|null}
    """
    db = asos.load_asos_forecasts_db()
    runs = db.get("runs", {})
    models = ["gfs", "aifs", "ifs", "nws"]
    now = datetime.now(timezone.utc)

    result = {}
    for sid in station_ids:
        station_obs = db.get("observations", {}).get(sid, {})
        result[sid] = {}
        for lead_time in lead_times:
            records = []
            for run_key, run_data in runs.items():
                try:
                    init_time = datetime.fromisoformat(run_key)
                    if init_time.tzinfo is None:
                        init_time = init_time.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

                forecast_hours = run_data.get("forecast_hours", [])
                if lead_time not in forecast_hours:
                    continue
                idx = forecast_hours.index(lead_time)

                valid_time = init_time + timedelta(hours=lead_time)
                if valid_time >= now:
                    continue

                def nearest_var(var_key, max_delta_min=90, _st=station_obs, _vt=valid_time):
                    best_val, best_delta = None, timedelta(minutes=max_delta_min + 1)
                    for ts_str, obs_data in _st.items():
                        try:
                            obs_time = datetime.fromisoformat(ts_str)
                            if obs_time.tzinfo is None:
                                obs_time = obs_time.replace(tzinfo=timezone.utc)
                        except Exception:
                            continue
                        delta = abs(obs_time - _vt)
                        if delta > timedelta(minutes=max_delta_min):
                            continue
                        val = obs_data.get(var_key)
                        if val is not None and delta < best_delta:
                            best_delta, best_val = delta, val
                    return best_val

                obs_temp     = nearest_var('temp')
                obs_dewpoint = nearest_var('dewpoint')
                stored       = asos.get_stored_observation(db, sid, valid_time, max_delta_minutes=70)
                obs_precip   = stored.get("precip_6hr") if stored else None

                tb, pb, db_bias = {}, {}, {}
                for m in models:
                    fcst = run_data.get(m, {}).get(sid)
                    if not fcst:
                        tb[m] = pb[m] = db_bias[m] = None
                        continue
                    ft  = fcst.get("temps",     [])
                    fp  = fcst.get("precips",   [])
                    fd  = fcst.get("dewpoints", [])
                    ftemp   = ft[idx]  if idx < len(ft)  else None
                    fprecip = fp[idx]  if idx < len(fp)  else None
                    fdew    = fd[idx]  if idx < len(fd)  else None
                    tb[m]      = round(ftemp   - obs_temp,     2) if ftemp   is not None and obs_temp     is not None else None
                    pb[m]      = round(fprecip - obs_precip,   3) if fprecip is not None and obs_precip   is not None else None
                    db_bias[m] = round(fdew    - obs_dewpoint, 2) if fdew    is not None and obs_dewpoint is not None else None

                if (obs_temp is None and obs_precip is None and obs_dewpoint is None
                        and all(v is None for v in tb.values())
                        and all(v is None for v in pb.values())
                        and all(v is None for v in db_bias.values())):
                    continue

                records.append({
                    't':  valid_time.astimezone(timezone.utc).isoformat(),
                    'ot': round(obs_temp,     1) if obs_temp     is not None else None,
                    'op': round(obs_precip,   3) if obs_precip   is not None else None,
                    'od': round(obs_dewpoint, 1) if obs_dewpoint is not None else None,
                    'tb': tb, 'pb': pb, 'db': db_bias,
                })

            records.sort(key=lambda r: r['t'])
            result[sid][str(lead_time)] = records

    return result


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_index_html(generated_at: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASOS Model Verification</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<style>
body {{ font-size: 0.875rem; }}
#map {{ height: 450px; background: #e8e8e8; }}
.leaflet-tooltip {{ background: white; border: 1px solid #ccc; border-radius: 4px; padding: 6px 10px; font-size: 13px; box-shadow: 0 2px 4px rgba(0,0,0,.15); }}
.chart-wrap {{ position: relative; height: 260px; }}
.chart-wrap-sm {{ position: relative; height: 200px; }}
th, td {{ white-space: nowrap; font-size: 0.78rem; padding: 0.25rem 0.4rem !important; }}
thead tr:first-child th {{ position: sticky; top: 0; background: white; z-index: 2; }}
thead tr:nth-child(2) th {{ position: sticky; top: 0; background: white; z-index: 1; }}
.color-scale-bar {{ width: 200px; height: 12px; border-radius: 4px; }}
.card-header {{ font-size: 0.85rem; }}
.form-select-sm {{ font-size: 0.8rem; }}
</style>
</head>
<body class="bg-light">

<div class="container-fluid py-3">

<div class="d-flex align-items-center justify-content-between flex-wrap gap-2 mb-3">
  <div>
    <h5 class="mb-0"><i class="bi bi-graph-up"></i> ASOS Model Verification</h5>
    <small class="text-muted">Updated {generated_at} &mdash; <span id="stationCount">loading...</span> &mdash; <a href="#" data-bs-toggle="modal" data-bs-target="#methodologyModal">Methodology</a></small>
  </div>
</div>

<!-- Methodology modal -->
<div class="modal fade" id="methodologyModal" tabindex="-1" aria-labelledby="methodologyModalLabel" aria-hidden="true">
  <div class="modal-dialog modal-lg modal-dialog-scrollable">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title" id="methodologyModalLabel"><i class="bi bi-info-circle"></i> Methodology</h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
      </div>
      <div class="modal-body">
        <h6>Data Source</h6>
        <p>Observations come from the <a href="https://mesonet.agron.iastate.edu/ASOS/" target="_blank">ASOS (Automated Surface Observing System)</a> network via Iowa Environmental Mesonet (IEM). Roughly 2,500 stations across the US report temperature, dew point, pressure, and precipitation.</p>

        <h6>Models</h6>
        <ul>
          <li><strong>GFS</strong> &mdash; NOAA Global Forecast System, 0.25° resolution, out to 15 days (360h)</li>
          <li><strong>AIFS</strong> &mdash; ECMWF AI Integrated Forecasting System, out to 15 days (360h)</li>
          <li><strong>IFS</strong> &mdash; ECMWF Integrated Forecasting System (open data), out to 15 days (360h)</li>
          <li><strong>NWS</strong> &mdash; National Weather Service point forecasts via the NWS API, limited to 7 days (168h)</li>
        </ul>

        <h6>Variables</h6>
        <ul>
          <li><strong>Temperature</strong> &mdash; 2-meter air temperature (°F). Matched to hourly METAR observations.</li>
          <li><strong>Dew Point</strong> &mdash; 2-meter dew point temperature (°F). Matched to hourly METAR observations.</li>
          <li><strong>Precipitation</strong> &mdash; 6-hour accumulated precipitation (in). Requires at least 4 of 6 hourly reports to be present; computed from the final per-hour METAR accumulation total.</li>
          <li><strong>Pressure</strong> &mdash; Altimeter setting (mb). Matched to automated 5-minute ASOS pressure observations.</li>
        </ul>

        <h6>Observation Matching</h6>
        <p>Each model grid point is bilinearly interpolated to the station location. For each forecast valid time, the nearest qualifying observation within a 30-minute window is used. Because ASOS stations report temperature and dew point at :56 past the hour (in METARs) and pressure every 5 minutes, a composite matching approach is used: the closest observation containing each variable is found independently, so no variable is penalized by the timing of another.</p>

        <h6>Metrics</h6>
        <ul>
          <li><strong>MAE</strong> (Mean Absolute Error) &mdash; average of |forecast &minus; observed| across all matched cases. Lower is better; always &ge; 0.</li>
          <li><strong>Bias</strong> &mdash; average of (forecast &minus; observed). Positive = model runs warm/wet/high; negative = model runs cold/dry/low.</li>
        </ul>

        <h6>Lead Times</h6>
        <p>Verification is computed at every 6-hour forecast lead from 6h to 360h (15 days). The map and station charts show a subset of these leads.</p>

        <h6>Averaging Periods</h6>
        <ul>
          <li><strong>All-time</strong> &mdash; cumulative stats since the system started collecting data. Forecast runs are kept for 21 days, but their error sums are folded into a running accumulator before deletion, so the all-time numbers grow indefinitely.</li>
          <li><strong>Last 20 days</strong> &mdash; rolling window limited to the most recent 20 days of runs.</li>
        </ul>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary btn-sm" data-bs-dismiss="modal">Close</button>
      </div>
    </div>
  </div>
</div>

<!-- Loading banner -->
<div id="loadingBanner" class="alert alert-info py-2">
  <div class="spinner-border spinner-border-sm me-2" role="status"></div>
  Loading verification data&hellip;
</div>

<!-- MAP CARD -->
<div class="card mb-3">
  <div class="card-header d-flex justify-content-between align-items-center flex-wrap gap-2">
    <span><i class="bi bi-geo-alt"></i> Station Map</span>
    <div class="d-flex gap-2 flex-wrap">
      <select class="form-select form-select-sm" id="selVar" style="width:auto">
        <option value="temp">Temperature</option>
        <option value="mslp">Pressure</option>
        <option value="precip">Precipitation</option>
        <option value="dewpoint">Dew Point</option>
      </select>
      <select class="form-select form-select-sm" id="selMetric" style="width:auto">
        <option value="mae">MAE</option>
        <option value="bias">Bias</option>
      </select>
      <select class="form-select form-select-sm" id="selModel" style="width:auto">
        <option value="gfs">GFS</option>
        <option value="aifs">AIFS</option>
        <option value="ifs">IFS</option>
        <option value="nws">NWS</option>
      </select>
      <select class="form-select form-select-sm" id="selLt" style="width:auto"></select>
      <div class="form-check form-switch align-self-center ms-1" id="mapPeriodWrap" style="display:none!important">
        <input class="form-check-input" type="checkbox" id="mapPeriodToggle">
        <label class="form-check-label small" for="mapPeriodToggle">Last 20 days</label>
      </div>
    </div>
  </div>
  <div class="card-body p-0">
    <div id="map"></div>
    <!-- Color scale -->
    <div class="d-flex align-items-center justify-content-center gap-3 py-2 border-top bg-light">
      <span id="scaleMin" class="text-muted small">0</span>
      <div id="scaleBar" class="color-scale-bar"></div>
      <span id="scaleMax" class="text-muted small">10</span>
      <span id="scaleUnits" class="text-muted small ms-1"></span>
    </div>
  </div>
</div>

<!-- STATION DETAIL (hidden until click) -->
<div id="stationDetail" class="card mb-3 d-none">
  <div class="card-header d-flex justify-content-between align-items-center flex-wrap gap-2">
    <span><i class="bi bi-pin-map"></i> <strong id="detailName"></strong></span>
    <div class="d-flex align-items-center gap-2">
      <div class="btn-group btn-group-sm" role="group">
        <input type="radio" class="btn-check" name="detailMetric" id="dm-mae" value="mae" autocomplete="off" checked>
        <label class="btn btn-outline-primary" for="dm-mae">MAE</label>
        <input type="radio" class="btn-check" name="detailMetric" id="dm-bias" value="bias" autocomplete="off">
        <label class="btn btn-outline-primary" for="dm-bias">Bias</label>
      </div>
      <button class="btn btn-sm btn-outline-secondary" onclick="closeDetail()"><i class="bi bi-x"></i></button>
    </div>
  </div>
  <div class="card-body">
    <p class="small text-muted mb-2">One line per model. Click legend items to show/hide.</p>
    <div class="row g-3">
      <div class="col-lg-6">
        <div class="card h-100">
          <div class="card-header"><i class="bi bi-thermometer-half"></i> Temperature</div>
          <div class="card-body"><div class="chart-wrap"><canvas id="dtTempChart"></canvas></div></div>
        </div>
      </div>
      <div class="col-lg-6">
        <div class="card h-100">
          <div class="card-header"><i class="bi bi-cloud-rain"></i> Precipitation</div>
          <div class="card-body"><div class="chart-wrap"><canvas id="dtPrecipChart"></canvas></div></div>
        </div>
      </div>
      <div class="col-lg-6">
        <div class="card h-100">
          <div class="card-header"><i class="bi bi-droplet-half"></i> Dew Point</div>
          <div class="card-body"><div class="chart-wrap"><canvas id="dtDewChart"></canvas></div></div>
        </div>
      </div>
      <div class="col-lg-6">
        <div class="card h-100">
          <div class="card-header"><i class="bi bi-speedometer"></i> Pressure</div>
          <div class="card-body"><div class="chart-wrap"><canvas id="dtMslpChart"></canvas></div></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- SPOTLIGHT OBS DETAIL (IAD, FME only) -->
<div id="obsDetail" class="card mb-3 d-none">
  <div class="card-header d-flex justify-content-between align-items-center flex-wrap gap-2">
    <span><i class="bi bi-clock-history"></i> Observations &amp; Bias &mdash; <strong id="obsDetailName"></strong></span>
    <div class="d-flex gap-2 align-items-center">
      <label class="form-label small text-muted mb-0">Lead:</label>
      <select class="form-select form-select-sm" id="obsLtSelect" style="width:auto">
        <option value="24" selected>24h</option>
        <option value="72">72h</option>
        <option value="168">7d</option>
      </select>
    </div>
  </div>
  <div class="card-body">
    <p class="small text-muted mb-2">Observed value (line) and model bias (bars, hover for details).
      <span class="ms-2">
        <span style="display:inline-block;width:10px;height:10px;background:#0d6efd;border-radius:2px"></span> GFS&nbsp;
        <span style="display:inline-block;width:10px;height:10px;background:#0dcaf0;border-radius:2px"></span> AIFS&nbsp;
        <span style="display:inline-block;width:10px;height:10px;background:#20c997;border-radius:2px"></span> IFS&nbsp;
        <span style="display:inline-block;width:10px;height:10px;background:#6c757d;border-radius:2px"></span> NWS
      </span>
    </p>
    <div class="row g-3">
      <div class="col-lg-6">
        <div class="card h-100">
          <div class="card-header"><i class="bi bi-thermometer-half"></i> Temperature</div>
          <div class="card-body">
            <div style="position:relative;height:180px"><canvas id="obsTempLineChart"></canvas></div>
            <div style="position:relative;height:150px;margin-top:8px"><canvas id="obsTempBiasChart"></canvas></div>
          </div>
        </div>
      </div>
      <div class="col-lg-6">
        <div class="card h-100">
          <div class="card-header"><i class="bi bi-droplet-half"></i> Dew Point</div>
          <div class="card-body">
            <div style="position:relative;height:180px"><canvas id="obsDewLineChart"></canvas></div>
            <div style="position:relative;height:150px;margin-top:8px"><canvas id="obsDewBiasChart"></canvas></div>
          </div>
        </div>
      </div>
      <div class="col-12">
        <div class="card">
          <div class="card-header"><i class="bi bi-cloud-rain"></i> Precipitation (6-hour)</div>
          <div class="card-body">
            <div style="position:relative;height:180px"><canvas id="obsPrecipLineChart"></canvas></div>
            <div style="position:relative;height:150px;margin-top:8px"><canvas id="obsPrecipBiasChart"></canvas></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- MEAN VERIFICATION TABLE -->
<div class="card mb-3">
  <div class="card-header d-flex align-items-center justify-content-between flex-wrap gap-2">
    <span><i class="bi bi-table"></i> Mean Verification — All Stations</span>
    <div class="d-flex gap-2 flex-wrap align-items-center">
      <div class="form-check form-switch mb-0 me-1">
        <input class="form-check-input" type="checkbox" id="periodToggle">
        <label class="form-check-label small" for="periodToggle">Last 20 days</label>
      </div>
      <div class="btn-group btn-group-sm">
        <input type="radio" class="btn-check" name="tableMetric" id="tm-mae" value="mae" autocomplete="off" checked>
        <label class="btn btn-outline-primary" for="tm-mae">MAE</label>
        <input type="radio" class="btn-check" name="tableMetric" id="tm-bias" value="bias" autocomplete="off">
        <label class="btn btn-outline-primary" for="tm-bias">Bias</label>
      </div>
      <div class="btn-group btn-group-sm" id="validHourFilter">
        <input type="radio" class="btn-check" name="validHour" id="vh-all" value="" autocomplete="off" checked>
        <label class="btn btn-outline-secondary" for="vh-all">All</label>
        <input type="radio" class="btn-check" name="validHour" id="vh-00" value="0" autocomplete="off">
        <label class="btn btn-outline-secondary" for="vh-00">00Z</label>
        <input type="radio" class="btn-check" name="validHour" id="vh-06" value="6" autocomplete="off">
        <label class="btn btn-outline-secondary" for="vh-06">06Z</label>
        <input type="radio" class="btn-check" name="validHour" id="vh-12" value="12" autocomplete="off">
        <label class="btn btn-outline-secondary" for="vh-12">12Z</label>
        <input type="radio" class="btn-check" name="validHour" id="vh-18" value="18" autocomplete="off">
        <label class="btn btn-outline-secondary" for="vh-18">18Z</label>
      </div>
    </div>
  </div>
  <div class="card-body p-0">
    <div class="table-responsive" style="max-height:60vh">
      <table class="table table-hover table-sm mb-0">
        <thead>
          <tr>
            <th rowspan="2">Lead</th><th rowspan="2">Runs</th>
            <th colspan="5" class="text-center border-start">Temp (°F)</th>
            <th colspan="5" class="text-center border-start">Precip (in)</th>
            <th colspan="5" class="text-center border-start">Dew Point (°F)</th>
            <th colspan="3" class="text-center border-start">Pressure (mb)</th>
          </tr>
          <tr>
            <th class="text-center border-start" style="background:rgba(13,110,253,.10)">GFS</th>
            <th class="text-center" style="background:rgba(13,202,240,.10)">AIFS</th>
            <th class="text-center" style="background:rgba(25,135,84,.10)">IFS</th>
            <th class="text-center" style="background:rgba(33,37,41,.10)">NWS</th>
            <th class="text-center">Win</th>
            <th class="text-center border-start" style="background:rgba(13,110,253,.10)">GFS</th>
            <th class="text-center" style="background:rgba(13,202,240,.10)">AIFS</th>
            <th class="text-center" style="background:rgba(25,135,84,.10)">IFS</th>
            <th class="text-center" style="background:rgba(33,37,41,.10)">NWS</th>
            <th class="text-center">Win</th>
            <th class="text-center border-start" style="background:rgba(13,110,253,.10)">GFS</th>
            <th class="text-center" style="background:rgba(13,202,240,.10)">AIFS</th>
            <th class="text-center" style="background:rgba(25,135,84,.10)">IFS</th>
            <th class="text-center" style="background:rgba(33,37,41,.10)">NWS</th>
            <th class="text-center">Win</th>
            <th class="text-center border-start" style="background:rgba(13,110,253,.10)">GFS</th>
            <th class="text-center" style="background:rgba(13,202,240,.10)">AIFS</th>
            <th class="text-center" style="background:rgba(25,135,84,.10)">IFS</th>
          </tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- TIME SERIES CHART -->
<div class="card mb-3">
  <div class="card-header d-flex justify-content-between align-items-center flex-wrap gap-2">
    <span><i class="bi bi-graph-up"></i> Verification Trends (Last 30 Days)</span>
    <div class="d-flex gap-2 flex-wrap">
      <select class="form-select form-select-sm" id="tsVar" style="width:auto">
        <option value="temp">Temperature</option>
        <option value="mslp">Pressure</option>
        <option value="precip">Precipitation</option>
        <option value="dewpoint">Dew Point</option>
      </select>
      <select class="form-select form-select-sm" id="tsMetric" style="width:auto">
        <option value="mae">MAE</option>
        <option value="bias">Bias</option>
      </select>
      <select class="form-select form-select-sm" id="tsLt" style="width:auto">
        <option value="6">6h</option>
        <option value="24" selected>24h</option>
        <option value="72">72h</option>
        <option value="168">7d</option>
      </select>
    </div>
  </div>
  <div class="card-body">
    <div style="position:relative;height:350px"><canvas id="tsChart"></canvas></div>
  </div>
</div>

</div><!-- /container-fluid -->

<script>
// ============================================================
// Data loaded async from sibling JSON files
// ============================================================
const MODELS    = ['gfs','aifs','ifs','nws'];
const MODEL_IDX = {{gfs:0,aifs:1,ifs:2,nws:3}};
const VAR_IDX   = {{temp:0,precip:1,dewpoint:2,mslp:3}};
const VAR_UNITS = {{temp:'°F',precip:'in',dewpoint:'°F',mslp:'mb'}};
const VAR_PRECIP = 'precip';
const SPOTLIGHT_STATIONS = new Set(['IAD','BWI']);

let MAP_DATA = null, TABLE_DATA = null, TS_DATA = null;
let asosMap = null, markers = [];
let dtCharts = {{}};
let tsChart = null;
let detailSt = null, detailIsMonthly = false;
let OBS_DATA = null;
let obsCharts = {{}};
let obsSid = null;

async function loadAll() {{
  const [m, t, ts, obs] = await Promise.all([
    fetch('data/map.json').then(r => r.json()),
    fetch('data/table.json').then(r => r.json()),
    fetch('data/timeseries.json').then(r => r.json()),
    fetch('data/station_obs.json?v={generated_at.replace(" ","_")}').then(r => r.json()).catch(() => null),
  ]);
  MAP_DATA = m; TABLE_DATA = t; TS_DATA = ts; OBS_DATA = obs;

  // Populate lead-time dropdown from published lead times
  const sel = document.getElementById('selLt');
  const ltLabels = {{6:'6h',12:'12h',24:'24h',48:'48h',72:'72h',120:'5d',168:'7d',240:'10d',288:'12d',360:'15d'}};
  MAP_DATA.lead_times.forEach(lt => {{
    const opt = document.createElement('option');
    opt.value = lt; opt.textContent = ltLabels[lt] || lt + 'h';
    if (lt === 24) opt.selected = true;
    sel.appendChild(opt);
  }});

  document.getElementById('stationCount').textContent =
    Object.keys(MAP_DATA.stations).length + ' stations';
  document.getElementById('loadingBanner').classList.add('d-none');

  // Show map period toggle only if monthly data was published
  if (MAP_DATA.monthly) {{
    document.getElementById('mapPeriodWrap').style.removeProperty('display');
  }}

  initMap();
  renderMap();
  renderTable();
  renderTs();
}}

// ============================================================
// Map
// ============================================================
function initMap() {{
  asosMap = L.map('map', {{center:[39.5,-98.5], zoom:4, minZoom:3, maxZoom:10}});
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution:'&copy; OpenStreetMap contributors', maxZoom:19
  }}).addTo(asosMap);
}}

function getMaeColor(value, variable) {{
  const max = variable === VAR_PRECIP ? 1 : 10;
  const r = Math.max(0, Math.min(1, value / max));
  if (r < 0.5) {{
    const t = r * 2;
    return `rgb(${{Math.round(34+t*200)}},${{Math.round(197-t*18)}},${{Math.round(94-t*86)}})`;
  }} else {{
    const t = (r-0.5)*2;
    return `rgb(${{Math.round(234+t*5)}},${{Math.round(179-t*111)}},${{Math.round(8+t*60)}})`;
  }}
}}
function getBiasColor(value, variable) {{
  const max = variable === VAR_PRECIP ? 1 : 10;
  const r = Math.max(-1, Math.min(1, value / max));
  if (variable === VAR_PRECIP) {{
    const g = (r+1)/2;
    if (g < 0.5) {{
      const t=g*2; return `rgb(${{Math.round(139+t*116)}},${{Math.round(69+t*186)}},${{Math.round(19+t*236)}})`;
    }} else {{
      const t=(g-0.5)*2; return `rgb(${{Math.round(255-t*221)}},${{Math.round(255-t*116)}},${{Math.round(255-t*221)}})`;
    }}
  }} else {{
    if (r < 0) {{
      const t=1+r; return `rgb(${{Math.round(59+t*196)}},${{Math.round(130+t*125)}},${{Math.round(246+t*9)}})`;
    }} else {{
      const t=r; return `rgb(255,${{Math.round(255-t*187)}},${{Math.round(255-t*187)}})`;
    }}
  }}
}}

function updateColorScale(variable, metric) {{
  const isP = variable === VAR_PRECIP;
  const bar = document.getElementById('scaleBar');
  const mn  = document.getElementById('scaleMin');
  const mx  = document.getElementById('scaleMax');
  document.getElementById('scaleUnits').textContent = VAR_UNITS[variable] || '';
  if (metric === 'bias') {{
    const lim = isP ? '±1.0' : '±10.0';
    mn.textContent = '-' + (isP ? '1.0' : '10.0');
    mx.textContent = isP ? '1.0' : '10.0';
    bar.style.background = isP
      ? 'linear-gradient(to right,rgb(139,69,19),white,rgb(34,139,34))'
      : 'linear-gradient(to right,#3b82f6,white 50%,#ef4444)';
  }} else {{
    mn.textContent = '0';
    mx.textContent = isP ? '1.0' : '10.0';
    bar.style.background = 'linear-gradient(to right,#22c55e,#eab308 50%,#ef4444)';
  }}
}}

function renderMap() {{
  if (!asosMap || !MAP_DATA) return;
  const variable = document.getElementById('selVar').value;
  const metric   = document.getElementById('selMetric').value;
  const model    = document.getElementById('selModel').value;
  const lt       = parseInt(document.getElementById('selLt').value, 10);
  const useMonthly = document.getElementById('mapPeriodToggle').checked && MAP_DATA.monthly;

  const stationSet = useMonthly ? MAP_DATA.monthly : MAP_DATA.stations;

  const mi = MODEL_IDX[model] ?? 0;
  const vi = VAR_IDX[variable] ?? 0;
  const ltIdx = MAP_DATA.lead_times.indexOf(lt);

  updateColorScale(variable, metric);
  markers.forEach(m => asosMap.removeLayer(m));
  markers = [];

  const maxScale = variable === VAR_PRECIP ? 1 : 10;

  for (const [sid, st] of Object.entries(stationSet)) {{
    const entry = st.d?.[mi]?.[vi]?.[ltIdx];
    if (!entry) continue;
    const [mae, bias] = entry;
    const value = metric === 'mae' ? mae : bias;
    if (value === null || value === undefined) continue;

    const color = metric === 'mae'
      ? getMaeColor(Math.abs(value), variable)
      : getBiasColor(value, variable);

    const norm = Math.min(Math.abs(metric === 'mae' ? value : value) / maxScale, 1);
    const radius = 3 + norm * 4.5;

    const marker = L.circleMarker([st.lat, st.lon], {{
      radius, fillColor: color, color:'#333', weight:1, opacity:0.8, fillOpacity:0.9
    }});

    const sign = bias >= 0 ? '+' : '';
    marker.bindTooltip(
      `<strong>${{st.name}}</strong> (${{sid}})<br>` +
      `MAE: ${{mae?.toFixed(2)}} ${{VAR_UNITS[variable]}}<br>` +
      `Bias: ${{sign}}${{bias?.toFixed(2)}} ${{VAR_UNITS[variable]}}`,
      {{direction:'top', offset:[0,-5]}}
    );
    marker.on('click', () => openDetail(sid, st, useMonthly));
    marker.addTo(asosMap);
    markers.push(marker);
  }}
}}

// ============================================================
// Station detail charts
// ============================================================
const DT_COLORS = {{
  gfs:  {{border:'#0d6efd', bg:'rgba(13,110,253,.12)'}},
  aifs: {{border:'#0dcaf0', bg:'rgba(13,202,240,.12)'}},
  ifs:  {{border:'#20c997', bg:'rgba(32,201,151,.12)'}},
  nws:  {{border:'#212529', bg:'rgba(33,37,41,.12)'}},
}};

function closeDetail() {{
  document.getElementById('stationDetail').classList.add('d-none');
  Object.values(dtCharts).forEach(c => c?.destroy());
  dtCharts = {{}};
  detailSt = null;
  closeObsDetail();
}}

function closeObsDetail() {{
  document.getElementById('obsDetail').classList.add('d-none');
  Object.values(obsCharts).forEach(c => c?.destroy());
  obsCharts = {{}};
  obsSid = null;
}}

function buildVarChart(canvasId, varKey, unit, metric) {{
  const vi = VAR_IDX[varKey];
  const st = detailSt;
  if (dtCharts[canvasId]) dtCharts[canvasId].destroy();
  if (!st) return;

  const isMae = metric === 'mae';
  const labels = MAP_DATA.lead_times.map(lt =>
    lt < 24 ? lt + 'h' : (lt % 24 === 0 ? (lt/24) + 'd' : lt + 'h'));

  const datasets = MODELS.map(m => {{
    const mi = MODEL_IDX[m];
    return {{
      label: m.toUpperCase(),
      data: MAP_DATA.lead_times.map((_,i) => st.d?.[mi]?.[vi]?.[i]?.[isMae ? 0 : 1] ?? null),
      borderColor: DT_COLORS[m].border,
      backgroundColor: 'transparent',
      borderWidth: 2,
      borderDash: isMae ? [] : [5, 4],
      tension: 0.3,
      pointRadius: 3,
    }};
  }});

  dtCharts[canvasId] = new Chart(document.getElementById(canvasId), {{
    type: 'line',
    data: {{labels, datasets}},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{mode:'index', intersect:false}},
      plugins: {{
        legend: {{position:'top', labels:{{usePointStyle:true, padding:12, font:{{size:11}}}}}},
        annotation: {{annotations: isMae ? {{}} : {{
          zero: {{type:'line', yMin:0, yMax:0, borderColor:'#aaa', borderWidth:1, borderDash:[5,5]}}
        }}}}
      }},
      scales: {{
        x: {{grid:{{display:false}}, title:{{display:true, text:'Lead Time'}}}},
        y: {{
          grid:{{color:'#e8e8e8'}},
          title:{{display:true, text:`${{isMae ? 'MAE' : 'Bias'}} (${{unit}})`}},
          min: isMae ? 0 : undefined,
        }},
      }}
    }}
  }});
}}

function renderDetailCharts() {{
  if (!detailSt) return;
  const metric = document.querySelector('input[name="detailMetric"]:checked').value;
  buildVarChart('dtTempChart',  'temp',     '°F', metric);
  buildVarChart('dtPrecipChart','precip',   'in',  metric);
  buildVarChart('dtDewChart',   'dewpoint', '°F', metric);
  buildVarChart('dtMslpChart',  'mslp',    'mb',  metric);
}}

function renderObsCharts(sid, lt) {{
  const records = OBS_DATA?.[sid]?.[String(lt)];
  if (!records || !records.length) return;

  const labels = records.map(r => {{
    const d = new Date(r.t);
    const md = d.toLocaleDateString('en-US', {{month:'numeric', day:'numeric'}});
    const hh = d.toLocaleTimeString('en-US', {{hour:'2-digit', hour12:false}});
    return `${{md}} ${{hh}}Z`;
  }});

  const mStyles = {{
    gfs:  {{bg:'#0d6efd'}},
    aifs: {{bg:'#0dcaf0'}},
    ifs:  {{bg:'#20c997'}},
    nws:  {{bg:'#6c757d'}},
  }};
  const mOrder = ['gfs','aifs','ifs','nws'];

  function buildBiasSets(key) {{
    const rankData = mOrder.map(() => []);
    const rankBg   = mOrder.map(() => []);
    const rankModel = mOrder.map(() => []);
    let hasAny = false;
    for (let i = 0; i < records.length; i++) {{
      const entries = [];
      for (const m of mOrder) {{
        const v = records[i][key]?.[m];
        if (v !== null && v !== undefined) entries.push({{m, v}});
      }}
      entries.sort((a,b) => Math.abs(a.v) - Math.abs(b.v));
      for (let r = 0; r < mOrder.length; r++) {{
        const e = entries[r];
        if (e) {{
          rankData[r].push(e.v);
          rankBg[r].push(mStyles[e.m].bg);
          rankModel[r].push(e.m);
          hasAny = true;
        }} else {{
          rankData[r].push(null);
          rankBg[r].push('rgba(0,0,0,0)');
          rankModel[r].push(null);
        }}
      }}
    }}
    if (!hasAny) return [];
    return mOrder.map((_, r) => {{
      if (!rankData[r].some(v => v !== null)) return null;
      return {{
        label: `Layer ${{r+1}}`,
        data: rankData[r],
        backgroundColor: rankBg[r],
        borderColor: rankBg[r],
        borderWidth: 1,
        grouped: false,
        barPercentage: 0.95,
        categoryPercentage: 1.0,
        modelByIndex: rankModel[r],
      }};
    }}).filter(Boolean);
  }}

  function biasTooltip(unit) {{
    return {{
      callbacks: {{
        label: ctx => {{
          const m = ctx.dataset.modelByIndex?.[ctx.dataIndex];
          const name = m ? m.toUpperCase() : 'Model';
          const v = ctx.parsed.y;
          if (v === null || v === undefined) return null;
          const dp = unit === 'in' ? 3 : 2;
          return `${{name}}: ${{v > 0 ? '+' : ''}}${{v.toFixed(dp)}} ${{unit}}`;
        }}
      }}
    }};
  }}

  const xScale = {{grid:{{display:false}}, ticks:{{maxRotation:45, minRotation:45, font:{{size:9}}}}}};

  function makeObs(id, type, datasets, yTitle, extraOpts) {{
    if (obsCharts[id]) obsCharts[id].destroy();
    const opts = {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{mode:'index', intersect:false}},
      plugins: {{legend:{{position:'top', labels:{{usePointStyle:true, padding:10, font:{{size:10}}}}}}}},
      scales: {{
        y: {{grid:{{color:'#e8e8e8'}}, title:{{display:true, text:yTitle}}}},
        x: xScale,
      }},
      ...extraOpts,
    }};
    obsCharts[id] = new Chart(document.getElementById(id), {{type, data:{{labels, datasets}}, options:opts}});
  }}

  function makeBias(id, datasets, yTitle, unit) {{
    if (obsCharts[id]) obsCharts[id].destroy();
    obsCharts[id] = new Chart(document.getElementById(id), {{
      type: 'bar',
      data: {{labels, datasets}},
      options: {{
        responsive: true, maintainAspectRatio: false,
        interaction: {{mode:'index', intersect:false}},
        plugins: {{legend:{{display:false}}, tooltip: biasTooltip(unit)}},
        scales: {{
          y: {{grid:{{color:'#e8e8e8'}}, title:{{display:true, text:yTitle}}}},
          x: xScale,
        }},
      }},
    }});
  }}

  // Temperature
  makeObs('obsTempLineChart', 'line',
    [{{label:'Temp (°F)', data:records.map(r => r.ot),
       borderColor:'#dc3545', backgroundColor:'rgba(220,53,69,.1)',
       borderWidth:2, tension:0.3, pointRadius:2, pointHoverRadius:5}}],
    'Temperature (°F)');
  makeBias('obsTempBiasChart', buildBiasSets('tb'), 'Bias (°F)', '°F');

  // Dew point
  makeObs('obsDewLineChart', 'line',
    [{{label:'Dew Point (°F)', data:records.map(r => r.od),
       borderColor:'#7c3aed', backgroundColor:'rgba(124,58,237,.1)',
       borderWidth:2, tension:0.3, pointRadius:2, pointHoverRadius:5}}],
    'Dew Point (°F)');
  makeBias('obsDewBiasChart', buildBiasSets('db'), 'Bias (°F)', '°F');

  // Precipitation
  makeObs('obsPrecipLineChart', 'line',
    [{{label:'6-hr Precip (in)', data:records.map(r => r.op),
       borderColor:'#20c997', backgroundColor:'rgba(32,201,151,.1)',
       borderWidth:2, tension:0.3, pointRadius:2, pointHoverRadius:5}}],
    'Precip (in)',
    {{plugins:{{legend:{{position:'top', labels:{{usePointStyle:true, padding:10, font:{{size:10}}}}}}}},
      scales:{{y:{{grid:{{color:'#e8e8e8'}}, title:{{display:true,text:'Precip (in)'}}, beginAtZero:true}}, x:xScale}}}});
  makeBias('obsPrecipBiasChart', buildBiasSets('pb'), 'Bias (in)', 'in');
}}

function openDetail(sid, st, isMonthly) {{
  detailSt = st;
  detailIsMonthly = isMonthly;
  document.getElementById('stationDetail').classList.remove('d-none');
  const periodLabel = isMonthly ? ' — last 20 days' : '';
  document.getElementById('detailName').textContent = st.name + ' (' + sid + ')' + periodLabel;
  try {{ renderDetailCharts(); }} catch(e) {{ console.error('renderDetailCharts failed:', e); }}
  if (SPOTLIGHT_STATIONS.has(sid)) {{
    if (OBS_DATA?.[sid]) {{
      obsSid = sid;
      document.getElementById('obsDetailName').textContent = st.name + ' (' + sid + ')';
      document.getElementById('obsDetail').classList.remove('d-none');
      renderObsCharts(sid, parseInt(document.getElementById('obsLtSelect').value, 10));
      setTimeout(() => document.getElementById('obsDetail').scrollIntoView({{behavior:'smooth', block:'start'}}), 50);
      return;
    }} else {{
      console.warn('Spotlight station', sid, 'but OBS_DATA is', OBS_DATA);
    }}
  }} else {{
    closeObsDetail();
  }}
  document.getElementById('stationDetail').scrollIntoView({{behavior:'smooth', block:'start'}});
}}

// ============================================================
// Mean verification table
// ============================================================
function fmtLt(lt) {{
  if (lt < 24) return lt + 'h';
  return (lt % 24 === 0) ? (lt/24)+'d' : (lt/24).toFixed(1)+'d';
}}
function pctMae(cur, base) {{
  if (cur == null || base == null || base === 0) return '';
  const p = Math.round((cur-base)/base*100);
  if (p === 0) return '';
  return ` <span style="color:${{p<0?'green':'red'}};font-size:.7em">${{p>0?'+':''}}${{p}}%</span>`;
}}
function winner(models) {{
  const v = models.filter(m => m.mae != null);
  if (!v.length) return ['--','badge bg-secondary'];
  const mn = Math.min(...v.map(m=>m.mae));
  const ws = v.filter(m=>m.mae===mn);
  return ws.length===1 ? [ws[0].n, ws[0].cls] : ['Tie','badge bg-secondary'];
}}

function renderTable() {{
  const period  = document.getElementById('periodToggle').checked ? 'monthly' : 'all';
  const vh      = document.querySelector('input[name="validHour"]:checked').value;
  const isMae   = document.querySelector('input[name="tableMetric"]:checked').value === 'mae';
  const key     = `${{period}}_${{vh===''?'all':vh}}`;
  const v       = TABLE_DATA?.data?.[key];
  const a       = TABLE_DATA?.data?.['all_all'];
  const isMonthly = TABLE_DATA?.use_monthly && period === 'monthly';
  const tbody   = document.getElementById('tbody');
  tbody.innerHTML = '';

  if (!v?.lead_times?.length) {{
    tbody.innerHTML = '<tr><td colspan="20" class="text-center text-muted">No data</td></tr>';
    return;
  }}

  const f1 = (val,ref) => val!=null ? val.toFixed(1) + (isMae && isMonthly ? pctMae(val,ref) : '') : '--';
  const f2 = (val,ref) => val!=null ? val.toFixed(2) + (isMae && isMonthly ? pctMae(val,ref) : '') : '--';
  const sg  = (val,dp=1) => val!=null ? (val>0?'+':'') + val.toFixed(dp) : '--';

  for (let i=0; i<v.lead_times.length; i++) {{
    const lt   = v.lead_times[i];
    const runs = Math.max(v.gfs_run_count?.[i]??0,v.aifs_run_count?.[i]??0,
                          v.ifs_run_count?.[i]??0, v.nws_run_count?.[i]??0);
    const gTm=v.gfs_temp_mae?.[i],  gTb=v.gfs_temp_bias?.[i];
    const aTm=v.aifs_temp_mae?.[i], aTb=v.aifs_temp_bias?.[i];
    const iTm=v.ifs_temp_mae?.[i],  iTb=v.ifs_temp_bias?.[i];
    const nTm=v.nws_temp_mae?.[i],  nTb=v.nws_temp_bias?.[i];
    const gRm=v.gfs_precip_mae?.[i],   gRb=v.gfs_precip_bias?.[i];
    const aRm=v.aifs_precip_mae?.[i],  aRb=v.aifs_precip_bias?.[i];
    const iRm=v.ifs_precip_mae?.[i],   iRb=v.ifs_precip_bias?.[i];
    const nRm=v.nws_precip_mae?.[i],   nRb=v.nws_precip_bias?.[i];
    const gDm=v.gfs_dewpoint_mae?.[i], gDb=v.gfs_dewpoint_bias?.[i];
    const aDm=v.aifs_dewpoint_mae?.[i],aDb=v.aifs_dewpoint_bias?.[i];
    const iDm=v.ifs_dewpoint_mae?.[i], iDb=v.ifs_dewpoint_bias?.[i];
    const nDm=v.nws_dewpoint_mae?.[i], nDb=v.nws_dewpoint_bias?.[i];
    const gPm=v.gfs_mslp_mae?.[i],     gPb=v.gfs_mslp_bias?.[i];
    const aPm=v.aifs_mslp_mae?.[i],    aPb=v.aifs_mslp_bias?.[i];
    const iPm=v.ifs_mslp_mae?.[i],     iPb=v.ifs_mslp_bias?.[i];

    const [tW,tC]=winner([{{n:'GFS',mae:gTm,cls:'badge bg-primary'}},{{n:'AIFS',mae:aTm,cls:'badge bg-info text-dark'}},{{n:'IFS',mae:iTm,cls:'badge bg-success'}},{{n:'NWS',mae:nTm,cls:'badge bg-dark'}}]);
    const [rW,rC]=winner([{{n:'GFS',mae:gRm,cls:'badge bg-primary'}},{{n:'AIFS',mae:aRm,cls:'badge bg-info text-dark'}},{{n:'IFS',mae:iRm,cls:'badge bg-success'}},{{n:'NWS',mae:nRm,cls:'badge bg-dark'}}]);
    const [dW,dC]=winner([{{n:'GFS',mae:gDm,cls:'badge bg-primary'}},{{n:'AIFS',mae:aDm,cls:'badge bg-info text-dark'}},{{n:'IFS',mae:iDm,cls:'badge bg-success'}},{{n:'NWS',mae:nDm,cls:'badge bg-dark'}}]);

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${{fmtLt(lt)}}</strong></td><td class="text-center">${{runs}}</td>
      <td class="text-center border-start">${{isMae?f1(gTm,a?.gfs_temp_mae?.[i]):sg(gTb)}}</td>
      <td class="text-center">${{isMae?f1(aTm,a?.aifs_temp_mae?.[i]):sg(aTb)}}</td>
      <td class="text-center">${{isMae?f1(iTm,a?.ifs_temp_mae?.[i]):sg(iTb)}}</td>
      <td class="text-center">${{isMae?f1(nTm,a?.nws_temp_mae?.[i]):sg(nTb)}}</td>
      <td><span class="${{tC}}">${{tW}}</span></td>
      <td class="text-center border-start">${{isMae?f2(gRm,a?.gfs_precip_mae?.[i]):sg(gRb,2)}}</td>
      <td class="text-center">${{isMae?f2(aRm,a?.aifs_precip_mae?.[i]):sg(aRb,2)}}</td>
      <td class="text-center">${{isMae?f2(iRm,a?.ifs_precip_mae?.[i]):sg(iRb,2)}}</td>
      <td class="text-center">${{isMae?f2(nRm,a?.nws_precip_mae?.[i]):sg(nRb,2)}}</td>
      <td><span class="${{rC}}">${{rW}}</span></td>
      <td class="text-center border-start">${{isMae?f1(gDm,a?.gfs_dewpoint_mae?.[i]):sg(gDb)}}</td>
      <td class="text-center">${{isMae?f1(aDm,a?.aifs_dewpoint_mae?.[i]):sg(aDb)}}</td>
      <td class="text-center">${{isMae?f1(iDm,a?.ifs_dewpoint_mae?.[i]):sg(iDb)}}</td>
      <td class="text-center">${{isMae?f1(nDm,a?.nws_dewpoint_mae?.[i]):sg(nDb)}}</td>
      <td><span class="${{dC}}">${{dW}}</span></td>
      <td class="text-center border-start">${{isMae?f1(gPm,a?.gfs_mslp_mae?.[i]):sg(gPb)}}</td>
      <td class="text-center">${{isMae?f1(aPm,a?.aifs_mslp_mae?.[i]):sg(aPb)}}</td>
      <td class="text-center">${{isMae?f1(iPm,a?.ifs_mslp_mae?.[i]):sg(iPb)}}</td>
    `;
    tbody.appendChild(tr);
  }}
}}

// ============================================================
// Time series chart
// ============================================================
function renderTs() {{
  const variable = document.getElementById('tsVar').value;
  const metric   = document.getElementById('tsMetric').value;
  const lt       = document.getElementById('tsLt').value;
  const combo    = TS_DATA?.[variable]?.[lt];
  const ctx      = document.getElementById('tsChart');

  if (tsChart) {{ tsChart.destroy(); tsChart = null; }}
  if (!combo) return;

  const labels = combo.dates.map(d => {{
    const dt = new Date(d);
    return dt.toLocaleDateString('en-US', {{month:'short', day:'numeric'}});
  }});

  const unit = VAR_UNITS[variable] || '';
  const mColors = {{
    gfs: {{border:'#0d6efd',bg:'rgba(13,110,253,.1)'}},
    aifs:{{border:'#0dcaf0',bg:'rgba(13,202,240,.1)'}},
    ifs: {{border:'#20c997',bg:'rgba(32,201,151,.1)'}},
    nws: {{border:'#212529',bg:'rgba(33,37,41,.1)'}},
  }};
  const datasets = ['gfs','aifs','ifs','nws']
    .filter(m => combo[m])
    .map(m => ({{
      label: m.toUpperCase(),
      data: combo[m][metric],
      borderColor: mColors[m].border,
      backgroundColor: mColors[m].bg,
      borderWidth: 2,
      tension: 0.3,
      pointRadius: 1,
      pointHoverRadius: 4,
    }}));

  tsChart = new Chart(ctx, {{
    type: 'line',
    data: {{labels, datasets}},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{mode:'index', intersect:false}},
      plugins: {{
        legend: {{position:'top', labels:{{usePointStyle:true, padding:16}}}},
        tooltip: {{
          callbacks: {{
            label: ctx => ctx.parsed.y != null
              ? `${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(2)}} ${{unit}}`
              : null
          }}
        }}
      }},
      scales: {{
        y: {{grid:{{color:'#e0e0e0'}}, title:{{display:true, text:`${{metric.toUpperCase()}} (${{unit}})`}}}},
        x: {{grid:{{display:false}}, title:{{display:true, text:'Date'}},
             ticks:{{maxRotation:45, minRotation:45}}}}
      }}
    }}
  }});
}}

// ============================================================
// Event listeners
// ============================================================
['selVar','selMetric','selModel','selLt'].forEach(id =>
  document.getElementById(id).addEventListener('change', renderMap));
document.getElementById('mapPeriodToggle').addEventListener('change', renderMap);

document.getElementById('periodToggle').addEventListener('change', renderTable);
document.querySelectorAll('input[name="tableMetric"]').forEach(r => r.addEventListener('change', renderTable));
document.querySelectorAll('input[name="validHour"]').forEach(r => r.addEventListener('change', renderTable));
document.querySelectorAll('input[name="detailMetric"]').forEach(r => r.addEventListener('change', renderDetailCharts));

['tsVar','tsMetric','tsLt'].forEach(id =>
  document.getElementById(id).addEventListener('change', renderTs));

document.getElementById('obsLtSelect').addEventListener('change', () => {{
  if (obsSid && OBS_DATA?.[obsSid]) {{
    renderObsCharts(obsSid, parseInt(document.getElementById('obsLtSelect').value, 10));
  }}
}});

// Fix two-row sticky thead: offset second row by first row's rendered height
function fixStickyHeader() {{
  const row1 = document.querySelector('thead tr:first-child');
  if (!row1) return;
  const h = row1.getBoundingClientRect().height;
  document.querySelectorAll('thead tr:nth-child(2) th').forEach(th => {{
    th.style.top = h + 'px';
  }});
}}

// Boot
loadAll().then(fixStickyHeader);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Site generation + push
# ---------------------------------------------------------------------------

def generate_site(output_dir: Path = PAGES_WORKTREE) -> None:
    output_dir = Path(output_dir)
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    print("Extracting station/map data from verification cache...")
    station_data = extract_station_data()
    n_stations = len(station_data['stations'])
    print(f"  {n_stations} stations with all-time verification data")

    print("Extracting mean verification table data...")
    table_data, use_monthly = evt.fetch_all_data(rebuild_monthly=False)

    if use_monthly:
        print("Extracting monthly (last 20 days) station data...")
        monthly_stations = extract_monthly_station_data(station_data['lead_times'])
        station_data['monthly'] = monthly_stations
        print(f"  {len(monthly_stations)} stations with monthly data")
    else:
        station_data['monthly'] = None

    print("Extracting time series data...")
    ts_data = extract_timeseries_data()

    # Write data files (compact JSON — no spaces)
    sep = (',', ':')
    with open(data_dir / 'map.json', 'w') as f:
        json.dump(station_data, f, separators=sep)
    with open(data_dir / 'table.json', 'w') as f:
        json.dump({'data': table_data, 'use_monthly': use_monthly,
                   'generated_at': generated_at}, f, separators=sep)
    with open(data_dir / 'timeseries.json', 'w') as f:
        json.dump(ts_data, f, separators=sep)

    print("Extracting spotlight station obs data (IAD, FME)...")
    spotlight_data = extract_spotlight_obs(SPOTLIGHT_STATIONS, SPOTLIGHT_LEAD_TIMES)
    with open(data_dir / 'station_obs.json', 'w') as f:
        json.dump(spotlight_data, f, separators=sep)

    # Write HTML
    with open(output_dir / 'index.html', 'w') as f:
        f.write(generate_index_html(generated_at))

    # Print sizes
    for fname in ['index.html', 'data/map.json', 'data/table.json', 'data/timeseries.json', 'data/station_obs.json']:
        p = output_dir / fname
        if p.exists():
            print(f"  {fname}: {p.stat().st_size / 1024:.0f} KB")

    print(f"Site written to {output_dir}")


def push_to_github_pages(output_dir: Path = PAGES_WORKTREE) -> bool:
    """Commit any changes and push to origin/gh-pages. Returns True if pushed."""
    output_dir = Path(output_dir)
    generated_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    subprocess.run(['git', '-C', str(output_dir), 'add', '-A'], check=True)

    # Check if there are staged changes
    result = subprocess.run(
        ['git', '-C', str(output_dir), 'diff', '--cached', '--quiet'])
    if result.returncode == 0:
        print("GitHub Pages: no changes to push")
        return False

    subprocess.run(
        ['git', '-C', str(output_dir), 'commit',
         '-m', f'Update: {generated_at}'],
        check=True
    )
    subprocess.run(
        ['git', '-C', str(output_dir), 'push', 'origin', 'gh-pages'],
        check=True
    )
    print(f"Pushed to gh-pages ({generated_at})")
    return True


def setup_pages_worktree() -> bool:
    """
    Ensure the gh-pages worktree exists at PAGES_WORKTREE.
    Idempotent — safe to call every sync.
    """
    if PAGES_WORKTREE.exists() and (PAGES_WORKTREE / '.git').exists():
        return True
    repo = Path(__file__).parent.parent / 'Weather_Models'  # /Users/kennypratt/Weather_Models
    try:
        subprocess.run(
            ['git', '-C', str(repo), 'worktree', 'add',
             str(PAGES_WORKTREE), 'gh-pages'],
            check=True, capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: could not set up gh-pages worktree: {e.stderr.decode()}")
        return False


if __name__ == '__main__':
    push = '--push' in sys.argv
    generate_site()
    if push:
        push_to_github_pages()
