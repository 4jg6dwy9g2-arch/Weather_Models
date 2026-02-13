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

# Create log file
log_file = os.path.join(log_dir, f"sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

with open(log_file, 'w') as f:
    f.write("=" * 50 + "\n")
    f.write(f"Sync started at: {datetime.now()}\n")
    f.write("=" * 50 + "\n\n")

    try:
        from app import run_master_sync
        results = run_master_sync()
        f.write(f"Master sync results:\\n{results}\\n")
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

# Cleanup old logs (keep last 48 hours)
try:
    for filename in os.listdir(log_dir):
        if filename.startswith('sync_') and filename.endswith('.log'):
            filepath = os.path.join(log_dir, filename)
            if os.path.getmtime(filepath) < (datetime.now().timestamp() - 48 * 3600):
                os.remove(filepath)
except Exception as e:
    print(f"Error cleaning up logs: {e}", file=sys.stderr)

sys.exit(0)
