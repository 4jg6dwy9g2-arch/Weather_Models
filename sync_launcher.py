#!/usr/bin/env python3
"""
Launch script for weather sync - called by launchd
"""
import os
import sys
import subprocess
from datetime import datetime

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

    # Run standalone sync script
    try:
        result = subprocess.run(
            [sys.executable, 'sync_standalone.py'],
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hours max
            cwd=script_dir
        )
        f.write(result.stdout)
        if result.stderr:
            f.write(result.stderr)

        if result.returncode != 0:
            f.write(f"\nSync script exited with code {result.returncode}\n")
    except subprocess.TimeoutExpired:
        f.write("Error: Sync timed out after 2 hours\n")
    except Exception as e:
        f.write(f"Error running sync script: {e}\n")

    f.write("\n" + "-" * 50 + "\n")
    f.write("Updating observations for existing runs...\n")
    f.write("-" * 50 + "\n\n")

    # Update observations
    try:
        result = subprocess.run(
            [sys.executable, 'update_observations.py'],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=script_dir
        )
        f.write(result.stdout)
        if result.stderr:
            f.write(result.stderr)
    except Exception as e:
        f.write(f"Error updating observations: {e}\n")

    f.write("\n" + "=" * 50 + "\n")
    f.write(f"Sync completed at: {datetime.now()}\n")
    f.write("=" * 50 + "\n")

# Cleanup old logs (keep last 30 days)
try:
    for filename in os.listdir(log_dir):
        if filename.startswith('sync_') and filename.endswith('.log'):
            filepath = os.path.join(log_dir, filename)
            if os.path.getmtime(filepath) < (datetime.now().timestamp() - 30 * 86400):
                os.remove(filepath)
except Exception as e:
    print(f"Error cleaning up logs: {e}", file=sys.stderr)

sys.exit(0)
