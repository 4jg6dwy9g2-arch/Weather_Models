#!/usr/bin/env python3
"""
Generate a self-contained HTML file of the ASOS verification table.
No server or data files needed to view — just open in a browser.
"""
import json
import sys
import os
from datetime import datetime, timezone

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import asos

PERIODS = ['all', 'monthly']
VALID_HOURS = [None, 0, 6, 12, 18]
MONTHLY_DAYS_BACK = 20

def fetch_all_data(rebuild_monthly: bool = True):
    if rebuild_monthly:
        print("  Rebuilding monthly cache (last 20 days)...")
        asos.rebuild_monthly_station_cache(days_back=MONTHLY_DAYS_BACK)
        print("  Rebuilding verification cache...")
        asos.precompute_verification_cache()
    db = asos.load_asos_forecasts_db()
    run_times = []
    for k in db.get("runs", {}):
        try:
            dt = datetime.fromisoformat(k)
            run_times.append(dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc))
        except Exception:
            pass
    span_days = int((max(run_times) - min(run_times)).total_seconds() / 86400) if len(run_times) >= 2 else 0
    use_monthly = span_days >= MONTHLY_DAYS_BACK
    print(f"  Data span: {span_days} days — monthly active: {use_monthly}")
    data = {}
    for period in PERIODS:
        for vh in VALID_HOURS:
            key = f"{period}_{'all' if vh is None else vh}"
            print(f"  Fetching {key}...")

            if period == 'monthly' and use_monthly:
                gfs   = asos.get_mean_verification_from_monthly_cache('gfs',  valid_hour=vh)
                aifs  = asos.get_mean_verification_from_monthly_cache('aifs', valid_hour=vh)
                ifs   = asos.get_mean_verification_from_monthly_cache('ifs',  valid_hour=vh)
                nws   = asos.get_mean_verification_from_monthly_cache('nws',  valid_hour=vh)
            else:
                gfs   = asos.get_mean_verification_from_cache('gfs',  valid_hour=vh)
                aifs  = asos.get_mean_verification_from_cache('aifs', valid_hour=vh)
                ifs   = asos.get_mean_verification_from_cache('ifs',  valid_hour=vh)
                nws   = asos.get_mean_verification_from_cache('nws',  valid_hour=vh)

            effective_period = period if (period != 'monthly' or use_monthly) else 'all'
            gfs_rc  = asos.get_run_counts_by_lead_time('gfs',  effective_period, valid_hour=vh)
            aifs_rc = asos.get_run_counts_by_lead_time('aifs', effective_period, valid_hour=vh)
            ifs_rc  = asos.get_run_counts_by_lead_time('ifs',  effective_period, valid_hour=vh)
            nws_rc  = asos.get_run_counts_by_lead_time('nws',  effective_period, valid_hour=vh)

            lead_times = gfs.get('lead_times', [])

            verification = {
                'lead_times':       lead_times,
                'gfs_run_count':    [gfs_rc.get(int(lt), 0)  for lt in lead_times],
                'aifs_run_count':   [aifs_rc.get(int(lt), 0) for lt in lead_times],
                'ifs_run_count':    [ifs_rc.get(int(lt), 0)  for lt in lead_times],
                'nws_run_count':    [nws_rc.get(int(lt), 0)  for lt in lead_times],
                'gfs_temp_mae':     gfs.get('temp_mae', []),
                'gfs_temp_bias':    gfs.get('temp_bias', []),
                'aifs_temp_mae':    aifs.get('temp_mae', []),
                'aifs_temp_bias':   aifs.get('temp_bias', []),
                'ifs_temp_mae':     ifs.get('temp_mae', []),
                'ifs_temp_bias':    ifs.get('temp_bias', []),
                'nws_temp_mae':     nws.get('temp_mae', []),
                'nws_temp_bias':    nws.get('temp_bias', []),
                'gfs_mslp_mae':     gfs.get('mslp_mae', []),
                'gfs_mslp_bias':    gfs.get('mslp_bias', []),
                'aifs_mslp_mae':    aifs.get('mslp_mae', []),
                'aifs_mslp_bias':   aifs.get('mslp_bias', []),
                'ifs_mslp_mae':     ifs.get('mslp_mae', []),
                'ifs_mslp_bias':    ifs.get('mslp_bias', []),
                'gfs_precip_mae':   gfs.get('precip_mae', []),
                'gfs_precip_bias':  gfs.get('precip_bias', []),
                'aifs_precip_mae':  aifs.get('precip_mae', []),
                'aifs_precip_bias': aifs.get('precip_bias', []),
                'ifs_precip_mae':   ifs.get('precip_mae', []),
                'ifs_precip_bias':  ifs.get('precip_bias', []),
                'nws_precip_mae':   nws.get('precip_mae', []),
                'nws_precip_bias':  nws.get('precip_bias', []),
            }
            data[key] = verification

    return data, use_monthly


def generate_html(data: dict, use_monthly: bool = False) -> str:
    generated_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    data_json = json.dumps(data)
    monthly_active_js = 'true' if use_monthly else 'false'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASOS Model Verification — {generated_at}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css">
<style>
  body {{ font-size: 0.875rem; }}
  th, td {{ white-space: nowrap; font-size: 0.78rem; padding: 0.25rem 0.4rem !important; }}
  .table-responsive {{ max-height: 80vh; overflow-y: auto; }}
  thead th {{ position: sticky; top: 0; background: white; z-index: 1; }}
  .bg-primary.bg-opacity-10 {{ background-color: rgba(13,110,253,0.10) !important; }}
  .bg-info.bg-opacity-10    {{ background-color: rgba(13,202,240,0.10) !important; }}
  .bg-success.bg-opacity-10 {{ background-color: rgba(25,135,84,0.10) !important; }}
  .bg-dark.bg-opacity-10    {{ background-color: rgba(33,37,41,0.10) !important; }}
</style>
</head>
<body class="p-3">

<div class="d-flex align-items-center justify-content-between flex-wrap gap-2 mb-3">
  <div>
    <h5 class="mb-0"><i class="bi bi-table"></i> ASOS Model Verification — All Stations</h5>
    <small class="text-muted">Generated {generated_at}</small>
  </div>
  <div class="d-flex gap-2 flex-wrap align-items-center">
    <div class="form-check form-switch mb-0">
      <input class="form-check-input" type="checkbox" id="periodToggle">
      <label class="form-check-label" for="periodToggle">Last 20 days</label>
    </div>
    <div class="btn-group btn-group-sm" role="group" id="validHourFilter">
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

<div class="table-responsive">
<table class="table table-hover table-sm mb-0">
<thead>
  <tr>
    <th>Lead</th><th>Runs</th>
    <th colspan="2" class="text-center bg-primary bg-opacity-10">GFS Temp (°F)</th>
    <th colspan="2" class="text-center bg-info bg-opacity-10 text-dark">AIFS Temp (°F)</th>
    <th colspan="2" class="text-center bg-success bg-opacity-10">IFS Temp (°F)</th>
    <th colspan="2" class="text-center bg-dark bg-opacity-10">NWS Temp (°F)</th>
    <th>Win</th>
    <th colspan="2" class="text-center bg-primary bg-opacity-10">GFS Pres (mb)</th>
    <th colspan="2" class="text-center bg-info bg-opacity-10 text-dark">AIFS Pres (mb)</th>
    <th colspan="2" class="text-center bg-success bg-opacity-10">IFS Pres (mb)</th>
    <th colspan="2" class="text-center bg-primary bg-opacity-10">GFS Precip (in)</th>
    <th colspan="2" class="text-center bg-info bg-opacity-10 text-dark">AIFS Precip (in)</th>
    <th colspan="2" class="text-center bg-success bg-opacity-10">IFS Precip (in)</th>
    <th colspan="2" class="text-center bg-dark bg-opacity-10">NWS Precip (in)</th>
    <th>Win</th>
  </tr>
  <tr>
    <th></th><th></th>
    <th class="text-center">MAE</th><th class="text-center">Bias</th>
    <th class="text-center">MAE</th><th class="text-center">Bias</th>
    <th class="text-center">MAE</th><th class="text-center">Bias</th>
    <th class="text-center">MAE</th><th class="text-center">Bias</th>
    <th></th>
    <th class="text-center">MAE</th><th class="text-center">Bias</th>
    <th class="text-center">MAE</th><th class="text-center">Bias</th>
    <th class="text-center">MAE</th><th class="text-center">Bias</th>
    <th class="text-center">MAE</th><th class="text-center">Bias</th>
    <th class="text-center">MAE</th><th class="text-center">Bias</th>
    <th class="text-center">MAE</th><th class="text-center">Bias</th>
    <th class="text-center">MAE</th><th class="text-center">Bias</th>
    <th></th>
  </tr>
</thead>
<tbody id="tbody"></tbody>
</table>
</div>

<script>
const ALL_DATA = {data_json};
const MONTHLY_ACTIVE = {monthly_active_js};

function formatLeadTime(lt) {{
  if (lt < 24) return lt + 'hr';
  const d = lt / 24;
  return d === Math.floor(d) ? Math.floor(d) + 'd' : d.toFixed(1) + 'd';
}}

function pctMae(cur, base) {{
  if (cur == null || base == null || base === 0) return '';
  const pct = Math.round((cur - base) / base * 100);
  if (pct === 0) return '';
  return ` <span style="color:${{pct < 0 ? 'green' : 'red'}};font-size:0.7em">${{pct > 0 ? '+' : ''}}${{pct}}%</span>`;
}}
function pctBias(cur, base) {{
  if (cur == null || base == null || base === 0) return '';
  const pct = Math.round((Math.abs(cur) - Math.abs(base)) / Math.abs(base) * 100);
  if (pct === 0) return '';
  return ` <span style="color:${{pct < 0 ? 'green' : 'red'}};font-size:0.7em">${{pct > 0 ? '+' : ''}}${{pct}}%</span>`;
}}

function renderTable() {{
  const period = document.getElementById('periodToggle').checked ? 'monthly' : 'all';
  const vh = document.querySelector('input[name="validHour"]:checked').value;
  const key = `${{period}}_${{vh === '' ? 'all' : vh}}`;
  const v = ALL_DATA[key];
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';

  if (!v || !v.lead_times || v.lead_times.length === 0) {{
    tbody.innerHTML = '<tr><td colspan="27" class="text-center text-muted">No data</td></tr>';
    return;
  }}

  const vAll = ALL_DATA[`all_all`];
  const a = (period === 'monthly' && MONTHLY_ACTIVE) ? vAll : null;

  for (let i = 0; i < v.lead_times.length; i++) {{
    const lt = v.lead_times[i];
    const runs = Math.max(v.gfs_run_count?.[i]??0, v.aifs_run_count?.[i]??0,
                          v.ifs_run_count?.[i]??0,  v.nws_run_count?.[i]??0);

    const gTm = v.gfs_temp_mae[i],   gTb = v.gfs_temp_bias[i];
    const aTm = v.aifs_temp_mae[i],  aTb = v.aifs_temp_bias[i];
    const iTm = v.ifs_temp_mae?.[i], iTb = v.ifs_temp_bias?.[i];
    const nTm = v.nws_temp_mae?.[i], nTb = v.nws_temp_bias?.[i];

    const gPm = v.gfs_mslp_mae[i],   gPb = v.gfs_mslp_bias[i];
    const aPm = v.aifs_mslp_mae[i],  aPb = v.aifs_mslp_bias[i];
    const iPm = v.ifs_mslp_mae?.[i], iPb = v.ifs_mslp_bias?.[i];

    const gRm = v.gfs_precip_mae?.[i],  gRb = v.gfs_precip_bias?.[i];
    const aRm = v.aifs_precip_mae?.[i], aRb = v.aifs_precip_bias?.[i];
    const iRm = v.ifs_precip_mae?.[i],  iRb = v.ifs_precip_bias?.[i];
    const nRm = v.nws_precip_mae?.[i],  nRb = v.nws_precip_bias?.[i];

    const tempModels = [
      {{n:'GFS',mae:gTm,cls:'badge bg-primary'}},
      {{n:'AIFS',mae:aTm,cls:'badge bg-info text-dark'}},
      {{n:'IFS',mae:iTm,cls:'badge bg-success'}},
      {{n:'NWS',mae:nTm,cls:'badge bg-dark'}}
    ].filter(m => m.mae != null);
    let tWin = '--', tCls = 'badge bg-secondary';
    if (tempModels.length) {{
      const mn = Math.min(...tempModels.map(m=>m.mae));
      const ws = tempModels.filter(m=>m.mae===mn);
      tWin = ws.length===1 ? ws[0].n : 'Tie';
      tCls = ws.length===1 ? ws[0].cls : 'badge bg-secondary';
    }}

    const precipModels = [
      {{n:'GFS',mae:gRm,cls:'badge bg-primary'}},
      {{n:'AIFS',mae:aRm,cls:'badge bg-info text-dark'}},
      {{n:'IFS',mae:iRm,cls:'badge bg-success'}},
      {{n:'NWS',mae:nRm,cls:'badge bg-dark'}}
    ].filter(m => m.mae != null);
    let pWin = '--', pCls = 'badge bg-secondary';
    if (precipModels.length) {{
      const mn = Math.min(...precipModels.map(m=>m.mae));
      const ws = precipModels.filter(m=>m.mae===mn);
      pWin = ws.length===1 ? ws[0].n : 'Tie';
      pCls = ws.length===1 ? ws[0].cls : 'badge bg-secondary';
    }}

    const f1 = (v,b,fn=1) => v!=null ? v.toFixed(fn)+(a?pctMae(v,b):'') : '--';
    const fb = (v,b,fn=1) => v!=null ? (v>0?'+':'')+v.toFixed(fn) : '--';
    const f2 = (v,b) => f1(v,b,2);
    const fb2 = (v,b) => fb(v,b,2);

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${{formatLeadTime(lt)}}</strong></td>
      <td class="text-center">${{runs}}</td>
      <td class="text-center">${{f1(gTm,a?.gfs_temp_mae?.[i])}}</td>
      <td class="text-center">${{fb(gTb,a?.gfs_temp_bias?.[i])}}</td>
      <td class="text-center">${{f1(aTm,a?.aifs_temp_mae?.[i])}}</td>
      <td class="text-center">${{fb(aTb,a?.aifs_temp_bias?.[i])}}</td>
      <td class="text-center">${{f1(iTm,a?.ifs_temp_mae?.[i])}}</td>
      <td class="text-center">${{fb(iTb,a?.ifs_temp_bias?.[i])}}</td>
      <td class="text-center">${{f1(nTm,a?.nws_temp_mae?.[i])}}</td>
      <td class="text-center">${{fb(nTb,a?.nws_temp_bias?.[i])}}</td>
      <td><span class="${{tCls}}">${{tWin}}</span></td>
      <td class="text-center">${{f1(gPm,a?.gfs_mslp_mae?.[i])}}</td>
      <td class="text-center">${{fb(gPb,a?.gfs_mslp_bias?.[i])}}</td>
      <td class="text-center">${{f1(aPm,a?.aifs_mslp_mae?.[i])}}</td>
      <td class="text-center">${{fb(aPb,a?.aifs_mslp_bias?.[i])}}</td>
      <td class="text-center">${{f1(iPm,a?.ifs_mslp_mae?.[i])}}</td>
      <td class="text-center">${{fb(iPb,a?.ifs_mslp_bias?.[i])}}</td>
      <td class="text-center">${{f2(gRm,a?.gfs_precip_mae?.[i])}}</td>
      <td class="text-center">${{fb2(gRb,a?.gfs_precip_bias?.[i])}}</td>
      <td class="text-center">${{f2(aRm,a?.aifs_precip_mae?.[i])}}</td>
      <td class="text-center">${{fb2(aRb,a?.aifs_precip_bias?.[i])}}</td>
      <td class="text-center">${{f2(iRm,a?.ifs_precip_mae?.[i])}}</td>
      <td class="text-center">${{fb2(iRb,a?.ifs_precip_bias?.[i])}}</td>
      <td class="text-center">${{f2(nRm,a?.nws_precip_mae?.[i])}}</td>
      <td class="text-center">${{fb2(nRb,a?.nws_precip_bias?.[i])}}</td>
      <td><span class="${{pCls}}">${{pWin}}</span></td>
    `;
    tbody.appendChild(tr);
  }}
}}

document.getElementById('periodToggle').addEventListener('change', renderTable);
document.querySelectorAll('input[name="validHour"]').forEach(r => r.addEventListener('change', renderTable));
renderTable();
</script>
</body>
</html>"""


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    no_rebuild = '--no-rebuild' in sys.argv
    out = args[0] if args else 'asos_verification_export.html'
    print("Fetching data...")
    data, use_monthly = fetch_all_data(rebuild_monthly=not no_rebuild)
    print("Generating HTML...")
    html = generate_html(data, use_monthly=use_monthly)
    with open(out, 'w') as f:
        f.write(html)
    size_kb = os.path.getsize(out) / 1024
    print(f"Written to {out} ({size_kb:.0f} KB)")
