#!/bin/bash

# Weather Model Sync Script
# Syncs Fairfax forecasts and ASOS station data

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/sync_$(date +%Y%m%d_%H%M%S).log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Log start
echo "======================================" >> "$LOG_FILE"
echo "Sync started at: $(date)" >> "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"

# Call the sync endpoint (assumes Flask app is running on localhost:5001)
curl -X GET "http://localhost:5001/api/sync-all" \
  -H "Content-Type: application/json" \
  -o - 2>&1 | tee -a "$LOG_FILE"

# Log completion
echo "" >> "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"
echo "Sync completed at: $(date)" >> "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Keep only last 30 days of logs
find "$LOG_DIR" -name "sync_*.log" -mtime +30 -delete

exit 0
