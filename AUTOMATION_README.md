# Weather Model Automation Setup

This guide will help you set up automatic syncing of weather forecast data every 6 hours.

## Overview

The automation consists of two parts:
1. **Flask App Service**: Keeps your web app running in the background
2. **Cron Job**: Automatically syncs weather data every 6 hours (1 AM, 7 AM, 1 PM, 7 PM)

---

## Part 1: Set Up Flask App as Auto-Start Service

This ensures the Flask app is always running so the cron job can sync data.

### Step-by-Step Instructions:

1. **Copy the launch agent:**
   ```bash
   cp /Users/kennypratt/Documents/Weather_Models/com.weather.app.plist ~/Library/LaunchAgents/
   ```

2. **Load and start the service:**
   ```bash
   launchctl load ~/Library/LaunchAgents/com.weather.app.plist
   launchctl start com.weather.app
   ```

3. **Verify it's running:**
   ```bash
   curl http://localhost:5000/
   ```
   You should see HTML from your dashboard.

4. **Check logs if needed:**
   ```bash
   tail -f /Users/kennypratt/Documents/Weather_Models/logs/flask_app.log
   ```

---

## Part 2: Set Up Automatic Sync Cron Job

This syncs all weather data (Fairfax + ASOS) every 6 hours.

### Step-by-Step Instructions:

1. **Open crontab editor:**
   ```bash
   crontab -e
   ```

2. **Add this line:**
   ```
   0 1,7,13,19 * * * /Users/kennypratt/Documents/Weather_Models/sync_weather.sh >> /Users/kennypratt/Documents/Weather_Models/logs/cron.log 2>&1
   ```

3. **Save and exit:**
   - In vim: Press `ESC`, then type `:wq` and press `ENTER`
   - In nano: Press `CTRL+X`, then `Y`, then `ENTER`

4. **Verify cron job was added:**
   ```bash
   crontab -l
   ```

### Sync Schedule:
- **1:00 AM** - Morning sync
- **7:00 AM** - Mid-morning sync
- **1:00 PM** - Afternoon sync
- **7:00 PM** - Evening sync

---

## Testing

### Test the sync script manually:
```bash
/Users/kennypratt/Documents/Weather_Models/sync_weather.sh
```

### View recent sync logs:
```bash
ls -lt /Users/kennypratt/Documents/Weather_Models/logs/sync_*.log | head -1 | awk '{print $9}' | xargs cat
```

### Monitor cron execution:
```bash
tail -f /Users/kennypratt/Documents/Weather_Models/logs/cron.log
```

---

## Managing Services

### Flask App Service:

**Stop:**
```bash
launchctl stop com.weather.app
```

**Restart:**
```bash
launchctl stop com.weather.app && launchctl start com.weather.app
```

**Disable auto-start:**
```bash
launchctl unload ~/Library/LaunchAgents/com.weather.app.plist
```

**Re-enable auto-start:**
```bash
launchctl load ~/Library/LaunchAgents/com.weather.app.plist
```

### Cron Job:

**Edit cron jobs:**
```bash
crontab -e
```

**Remove all cron jobs:**
```bash
crontab -r
```

**Temporarily disable a specific job:**
Add `#` at the beginning of the line in crontab

---

## Troubleshooting

### Cron Job Not Running:

1. **Check cron has Full Disk Access** (macOS Catalina+):
   - Go to System Preferences > Security & Privacy > Privacy > Full Disk Access
   - Add `/usr/sbin/cron` to the list

2. **Check the cron log:**
   ```bash
   cat /Users/kennypratt/Documents/Weather_Models/logs/cron.log
   ```

3. **Verify the script is executable:**
   ```bash
   ls -la /Users/kennypratt/Documents/Weather_Models/sync_weather.sh
   ```
   Should show `-rwxr-xr-x` (the `x` means executable)

### Flask App Not Starting:

1. **Check error logs:**
   ```bash
   cat /Users/kennypratt/Documents/Weather_Models/logs/flask_app_error.log
   ```

2. **Verify Python path:**
   ```bash
   which python3
   ```
   Update the path in `com.weather.app.plist` if different

3. **Test manually:**
   ```bash
   cd /Users/kennypratt/Documents/Weather_Models
   python3 app.py
   ```

### Sync Failing:

1. **Check if Flask app is running:**
   ```bash
   curl http://localhost:5000/api/sync-all
   ```

2. **Check sync logs for errors:**
   ```bash
   tail -50 /Users/kennypratt/Documents/Weather_Models/logs/sync_*.log
   ```

---

## Files Created

- `sync_weather.sh` - Sync script that calls the API
- `com.weather.app.plist` - LaunchAgent config for Flask app
- `setup_cron.txt` - Cron setup instructions
- `setup_service.txt` - Service setup instructions
- `AUTOMATION_README.md` - This file

## Log Files

All logs are stored in: `/Users/kennypratt/Documents/Weather_Models/logs/`

- `sync_YYYYMMDD_HHMMSS.log` - Individual sync run logs (kept 30 days)
- `cron.log` - Cron execution log
- `flask_app.log` - Flask application output
- `flask_app_error.log` - Flask application errors
