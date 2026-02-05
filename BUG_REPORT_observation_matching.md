# Bug Report: ASOS Verification Observation Matching

## Summary

**Issue**: ASOS verification was only matching ~1600 stations despite having 2920 stations with forecasts and 2557 stations with observations.

**Root Cause**: Bug in `get_cached_observation()` function in `asos.py` (line 1700-1703) introduced in commit da6a3e4.

**Impact**: ~928 stations (32% of all stations) were excluded from verification even though they had valid observations.

---

## Bug Details

### Location
File: `/Users/kennypratt/Documents/Weather_Models/asos.py`
Function: `precompute_verification_cache()` → `get_cached_observation()`
Lines: 1700-1703

### The Bug

When an exact time match was found, the function set `best_match` but **forgot to set `best_delta` to 0**:

```python
# BUGGY CODE (lines 1700-1703)
if target_time_str in station_cache:
    best_match = station_cache[target_time_str].copy()
else:
    # Find nearest within tolerance...
```

The problem:
1. `best_delta` was initialized to `timedelta(minutes=31)` on line 1697
2. When exact match found, `best_match` was set but `best_delta` remained at 31 minutes
3. Later check `if best_match and best_delta <= timedelta(minutes=30):` would fail (line 1715)
4. Function returned `None` even for exact matches!

### The Fix

Set `best_delta = timedelta(0)` when exact match is found:

```python
# FIXED CODE
if target_time_str in station_cache:
    best_match = station_cache[target_time_str].copy()
    best_delta = timedelta(0)  # Exact match has zero time delta
else:
    # Find nearest within tolerance...
```

---

## Investigation Process

### 1. Initial Analysis
- Observed discrepancy: 2920 forecasts, 2557 observations, but only ~1600 matched
- Checked for filtering logic, time matching issues, and data quality problems

### 2. Observation Data Quality Check
Created a debug script (since removed) to verify observation data:
- Found NO empty observations (filters from da6a3e4 working correctly)
- Found 67.4% of observations have MSLP-only (from altimeter conversion)
- Found 30.6% of observations have all three variables (temp, mslp, precip)
- Conclusion: Observation data is clean and valid

### 3. Time Matching Analysis
Created a debug script (since removed) to check temporal alignment:
- Found 2482 stations have observations within 30 minutes of F006 valid time
- Found distribution: most obs at :00, :15, :30, :45, :55 minutes
- Conclusion: Time matching should work for 2482 stations

### 4. Observation Lookup Trace
Created a debug script (since removed) to simulate exact matching logic:
- Confirmed function returns 1558 matches (bug reproduced)
- Found 999 stations with observations marked as "NO VALID OBSERVATIONS"
- Identified specific failing stations: 1M4, ANB, BHM (all had exact matches!)

### 5. Deep Dive on Station 1M4
Created a debug script (since removed) to trace step-by-step:
- Station 1M4 has 2471 observations
- Has EXACT match at target time 2026-02-05T12:00:00+00:00
- Found `best_match` was set correctly
- Found `best_delta` remained at 31 minutes (not updated!)
- Found final check `best_delta <= 30 minutes` failed
- **ROOT CAUSE IDENTIFIED**: Missing `best_delta = timedelta(0)` assignment

---

## Verification of Fix

### Before Fix
```
Total stations with GFS forecast: 2920
Stations with matched observation: 1558
Stations with no match: 1362
Match rate: 53.4%
```

### After Fix
```
Total stations with GFS forecast: 2920
Stations with matched observation: 2486
Stations with no match: 434
Match rate: 85.1%
```

### Improvement
- **+928 stations** now have verification data (+59% increase)
- **Match rate improved from 53.4% to 85.1%** (+31.7 percentage points)
- Nearly matches theoretical maximum of 2482 stations (99.8% of expected)

---

## Impact on Verification

### Temperature Verification
- Before: ~1444 stations
- After: ~2400 stations (estimated)
- Impact: +66% more stations

### MSLP Verification
- Before: ~1510 stations
- After: ~2450 stations (estimated)
- Impact: +62% more stations

### Precipitation Verification
- Before: Limited stations
- After: ~2400 stations at synoptic times (estimated)
- Impact: Significant improvement for 6hr precip totals

---

## Files Changed

1. `/Users/kennypratt/Documents/Weather_Models/asos.py` (line 1703)
   - Added: `best_delta = timedelta(0)` for exact matches

---

## Debug Scripts Created

Debug scripts created during investigation were removed after validation.

---

## Recommendations

1. **Immediate**: Regenerate verification cache after applying fix
   - Run sync to fetch new observations
   - Or manually call `precompute_verification_cache()`

2. **Testing**: Verify station counts in verification UI
   - Should see ~2400+ stations with verification data
   - Check that stations like 1M4, ANB, BHM now have data

3. **Future**: Add unit tests for observation matching
   - Test exact time match case
   - Test near time match case
   - Test edge cases (30 min boundary, 31 min rejection)

4. **Monitoring**: Log matching statistics during cache precomputation
   - Total stations processed
   - Stations matched (exact vs near)
   - Stations rejected (no obs, time too far, etc.)

---

## Lessons Learned

1. **When optimizing code, test edge cases thoroughly**
   - The exact match optimization was added for performance
   - But forgot to update all relevant state (`best_delta`)

2. **Validate assumptions with data**
   - Initial assumption: "observations are empty" → FALSE
   - Actual problem: "matching logic bug" → TRUE

3. **Step-by-step debugging is powerful**
   - Tracing execution revealed exact line where logic failed
   - Comparing expected vs actual state isolated the bug

4. **Create focused debug scripts**
   - Each script investigated one hypothesis
   - Incremental narrowing led to root cause

---

## Related Commits

- **da6a3e4**: "Optimize ASOS verification" - Introduced the bug
- **f0366e9**: "Use altimeter setting" - Not related to bug (red herring)
- **Current fix**: Resolves observation matching bug in da6a3e4
