#!/usr/bin/env python3
"""
Launch script for weather sync - called by launchd
"""
import os
import sys
from datetime import datetime
import traceback

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
    except Exception as e:
        f.write(f"Error running master sync: {e}\n")
        f.write(traceback.format_exc() + "\n")

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
