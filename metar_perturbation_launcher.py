#!/usr/bin/env python3
"""
Hourly launcher for METAR pressure perturbation cache rebuild.
Called by launchd every 3600 seconds.

1. Fetches fresh 5-min METAR altimeter data from IEM → asos_metar_pressure.json
2. Recomputes band-pass perturbation cache → asos_metar_perturbations.json

These files are completely separate from the verification database (asos_forecasts.json).
"""
import os
import sys
import json
from datetime import datetime
import traceback
import subprocess

IMESSAGE_RECIPIENT = os.environ.get("IMESSAGE_RECIPIENT", "")

# Change to script directory so imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

log_dir = os.path.join(script_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"metar_perturb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


def _send_imessage(recipient: str, message: str) -> None:
    if not recipient:
        return
    script = f'''
    tell application "Messages"
        set targetService to first service whose service type is iMessage
        set targetBuddy to buddy "{recipient}" of targetService
        send "{message}" to targetBuddy
    end tell
    '''
    try:
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True, text=True)
    except Exception:
        pass


with open(log_file, "w") as f:
    f.write("=" * 50 + "\n")
    f.write(f"METAR perturbation rebuild started at: {datetime.now()}\n")
    f.write("=" * 50 + "\n\n")

    try:
        import asos as asos_module
        from app import rebuild_asos_metar_perturbation_cache, ASOS_METAR_PERTURBATION_FILE

        # Step 1: Fetch fresh 5-min METAR pressure data from IEM
        f.write("Syncing METAR pressure data from IEM...\n")
        sync_result = asos_module.sync_asos_metar_pressure(lookback_hours=28)
        f.write(f"Sync result: {sync_result}\n")

        # Step 2: Rebuild perturbation cache from the dedicated pressure file
        f.write("Rebuilding perturbation cache...\n")
        payload = rebuild_asos_metar_perturbation_cache(hours=24)
        with open(ASOS_METAR_PERTURBATION_FILE, "w") as mf:
            json.dump(payload, mf)

        n_stations = len(payload.get("stations", []))
        latest = payload.get("latest_time", "unknown")
        f.write(f"Rebuilt: {n_stations} stations, latest obs at {latest}\n")

    except Exception as e:
        err_msg = f"METAR perturbation rebuild failed: {e}"
        f.write(err_msg + "\n")
        f.write(traceback.format_exc() + "\n")
        notify_msg = (
            f"METAR perturbation error on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"{err_msg}\n"
            f"Log: {log_file}"
        )
        _send_imessage(IMESSAGE_RECIPIENT, notify_msg)

    f.write("\n" + "=" * 50 + "\n")
    f.write(f"Completed at: {datetime.now()}\n")
    f.write("=" * 50 + "\n")

# Cleanup rotating METAR logs (keep only the newest one)
try:
    perturb_logs = sorted(
        [
            os.path.join(log_dir, fn)
            for fn in os.listdir(log_dir)
            if fn.startswith("metar_perturb_") and fn.endswith(".log")
        ],
        key=os.path.getmtime,
        reverse=True,
    )
    for old_file in perturb_logs[1:]:
        os.remove(old_file)
except Exception as e:
    print(f"Error cleaning up logs: {e}", file=sys.stderr)

sys.exit(0)
