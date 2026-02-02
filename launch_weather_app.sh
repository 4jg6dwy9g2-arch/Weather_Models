#!/bin/bash

# Change to the app directory
cd /Users/kennypratt/Documents/Weather_Models

# Check if Flask is already running
if lsof -i :5001 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Flask server already running"
else
    echo "Starting Flask server..."
    # Start Flask in background, completely detached
    nohup /Users/kennypratt/anaconda3/bin/python3 app.py > /tmp/weather_app.log 2>&1 &
    # Wait for server to start
    sleep 3
fi

# Open browser
open http://localhost:5001
