#!/usr/bin/env python3
"""
Launch script for weather sync - called by launchd
"""
import os
import sys
from datetime import datetime
import traceback
import subprocess

IMESSAGE_RECIPIENT = os.environ.get("IMESSAGE_RECIPIENT", "")


def _send_imessage(recipient: str, message: str) -> None:
    """Send an iMessage via Messages.app using osascript (macOS only)."""
    script = f'''
    tell application "Messages"
        set targetService to first service whose service type is iMessage
        set targetBuddy to buddy "{recipient}" of targetService
        send "{message}" to targetBuddy
    end tell
    '''
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        # Avoid raising from the notifier; log failure in the sync log
        pass


def _check_results(results: dict) -> list[str]:
    """
    Inspect sync results for soft failures and return a list of warning strings.
    Returns an empty list if everything looks healthy.
    """
    warnings = []

    if not isinstance(results, dict):
        return ["Sync returned unexpected result type"]

    if results.get("success") is False:
        warnings.append("Sync flagged success=False")

    for err in results.get("errors", []):
        warnings.append(f"Sync error: {err}")

    asos = results.get("asos", {})
    models = asos.get("models", {})

    for model in ("gfs", "aifs", "ifs", "nws"):
        m = models.get(model, {})
        status = m.get("status")
        stations = m.get("stations")
        if status and status != "synced":
            warnings.append(f"{model.upper()} status: {status}")
        if isinstance(stations, int) and stations == 0:
            warnings.append(f"{model.upper()} synced 0 stations")

    obs = models.get("observations", {})
    obs_status = obs.get("status")
    obs_count = obs.get("count")
    if obs_status and obs_status != "synced":
        warnings.append(f"Observations status: {obs_status}")
    if isinstance(obs_count, int) and obs_count == 0:
        warnings.append("Observations synced 0 records")

    fairfax = results.get("fairfax", {})
    if isinstance(fairfax, dict) and fairfax.get("status") not in ("synced", "current", None):
        warnings.append(f"Fairfax status: {fairfax.get('status')}")

    return warnings

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Create logs directory
log_dir = os.path.join(script_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Reset launchd stream logs each run so they do not grow indefinitely.
for _stream_name in ("launchd.log", "launchd_error.log"):
    try:
        with open(os.path.join(log_dir, _stream_name), "w"):
            pass
    except Exception:
        pass

# Create log file
log_file = os.path.join(log_dir, f"sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

with open(log_file, 'w') as f:
    f.write("=" * 50 + "\n")
    f.write(f"Sync started at: {datetime.now()}\n")
    f.write("=" * 50 + "\n\n")

    try:
        from app import run_master_sync
        import asos as asos_module
        results = run_master_sync()
        f.write(f"Master sync results:\\n{results}\\n")

        # Sync 1-min pressure data and rebuild perturbation cache overnight only
        if datetime.now().hour < 6:
            f.write("Overnight run detected — syncing 1-min ASOS pressure data...\\n")
            try:
                one_min_result = asos_module.sync_asos_1min_pressure(lookback_hours=72)
                f.write(f"1-min pressure sync: {one_min_result}\\n")
            except Exception as _e:
                f.write(f"1-min pressure sync failed: {_e}\\n")

            f.write("Rebuilding pressure perturbation cache...\\n")
            try:
                from app import build_and_append_asos_pressure_perturbation_history
                pp = build_and_append_asos_pressure_perturbation_history(
                    hours=24, cadence_minutes=5, short_minutes=10, long_minutes=180
                )
                f.write(f"Perturbation cache rebuilt: {pp}\\n")
            except Exception as _e:
                f.write(f"Perturbation cache rebuild failed: {_e}\\n")

        warnings = _check_results(results)
        if warnings:
            notify_msg = (
                f"Weather sync issues on {datetime.now().strftime('%Y-%m-%d %H:%M')}\\n"
                + "\\n".join(f"• {w}" for w in warnings)
                + f"\\nLog: {log_file}"
            )
            _send_imessage(IMESSAGE_RECIPIENT, notify_msg)
    except Exception as e:
        err_msg = f"Error running master sync: {e}"
        f.write(err_msg + "\n")
        f.write(traceback.format_exc() + "\n")
        notify_msg = (
            f"Weather sync error on {datetime.now().strftime('%Y-%m-%d %H:%M')}\\n"
            f"{err_msg}\\n"
            f"Log: {log_file}"
        )
        _send_imessage(IMESSAGE_RECIPIENT, notify_msg)

    f.write("\n" + "=" * 50 + "\n")
    f.write(f"Sync completed at: {datetime.now()}\n")
    f.write("=" * 50 + "\n")

# Cleanup rotating sync logs (keep only the newest one)
try:
    sync_logs = sorted(
        [
            os.path.join(log_dir, fn)
            for fn in os.listdir(log_dir)
            if fn.startswith("sync_") and fn.endswith(".log")
        ],
        key=os.path.getmtime,
        reverse=True,
    )
    for old_file in sync_logs[1:]:
        os.remove(old_file)
except Exception as e:
    print(f"Error cleaning up logs: {e}", file=sys.stderr)

sys.exit(0)
