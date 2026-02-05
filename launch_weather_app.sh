#!/bin/bash

# Change to the app directory
cd /Users/kennypratt/Documents/Weather_Models

# Check if Flask is already running
if lsof -i :5001 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Flask server already running"
    open http://localhost:5001
    exit 0
fi

echo "Starting Flask server..."

# Open the browser shortly after launch
(sleep 3; open http://localhost:5001) &

# Run Flask in the foreground so closing the terminal stops it
exec /Users/kennypratt/anaconda3/bin/python3 app.py > /tmp/weather_app.log 2>&1
