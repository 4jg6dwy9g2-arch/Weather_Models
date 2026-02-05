#!/bin/bash

# PanguWeather Flask App Startup Script

echo "=========================================="
echo "  PanguWeather Flask App"
echo "=========================================="
echo ""

# Check if model files exist
if [ ! -f "pangu_weather_24.onnx" ] || [ ! -f "pangu_weather_6.onnx" ]; then
    echo "⚠️  Warning: PanguWeather model files not found in current directory"
    echo "   Expected:"
    echo "   - pangu_weather_24.onnx"
    echo "   - pangu_weather_6.onnx"
    echo ""
    echo "   Make sure you're running from the ML_Weather_Models directory"
    echo ""
fi

# Create forecast directory
mkdir -p pangu_forecasts

# Start the Flask app
echo "Starting Flask app on http://localhost:5002"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python pangu_app.py
