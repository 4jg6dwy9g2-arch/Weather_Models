"""
Gravity wave structure detection from ASOS surface pressure perturbations.

Pipeline:
  1. Interpolate irregular station perturbations onto a regular lat/lon grid
  2. Gaussian-smooth to suppress station-spacing noise
  3. Detect connected positive/negative anomaly regions (potential wave packets)
  4. Estimate bulk propagation vector via 2D phase cross-correlation of
     consecutive frames
  5. Return contour paths for map overlay
"""
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, label
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Domain: CONUS + southern Canada + northern Mexico
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 22.0, 52.0
LON_MIN, LON_MAX = -127.0, -63.0
GRID_RES = 0.5          # degrees per cell
KM_PER_DEG_LAT = 111.0  # constant
# km per degree longitude at domain centre (~37°N)
_CENTRE_LAT = (LAT_MIN + LAT_MAX) / 2.0
KM_PER_DEG_LON = KM_PER_DEG_LAT * np.cos(np.radians(_CENTRE_LAT))


def make_grid():
    """Return (lats_1d, lons_1d, lats_2d, lons_2d) for the analysis domain."""
    lats = np.arange(LAT_MIN, LAT_MAX + GRID_RES, GRID_RES)
    lons = np.arange(LON_MIN, LON_MAX + GRID_RES, GRID_RES)
    grid_lons, grid_lats = np.meshgrid(lons, lats)
    return lats, lons, grid_lats, grid_lons


def interpolate_stations(station_lats, station_lons, station_vals,
                         grid_lats, grid_lons,
                         max_dist_km=250, min_neighbors=3,
                         max_nearest_km=150):
    """
    Linear Delaunay triangulation from irregular stations to the regular grid.

    Two masking passes:
      1. min_neighbors within max_dist_km — requires a cluster of stations,
         not just a single coastal airport.
      2. max_nearest_km — masks any cell whose *nearest* station is farther
         than this distance.  This is the key guard against Canada / ocean
         bleed-through: even a dense coastal cluster cannot paint ocean cells
         that are genuinely far from all stations.
    """
    pts  = np.column_stack([station_lons, station_lats])
    vals = np.asarray(station_vals, dtype=np.float64)
    grid = griddata(pts, vals, (grid_lons, grid_lats), method='linear')

    # KD-tree in approximate km (flat-Earth, good enough for masking)
    tree = cKDTree(np.column_stack([
        np.asarray(station_lons) * KM_PER_DEG_LON,
        np.asarray(station_lats) * KM_PER_DEG_LAT,
    ]))
    grid_pts_km = np.column_stack([
        grid_lons.ravel() * KM_PER_DEG_LON,
        grid_lats.ravel() * KM_PER_DEG_LAT,
    ])

    # Pass 1: need at least min_neighbors within max_dist_km
    counts = tree.query_ball_point(grid_pts_km, r=max_dist_km, return_length=True)
    too_sparse = counts.reshape(grid_lats.shape) < min_neighbors
    grid[too_sparse] = np.nan

    # Pass 2: nearest single station must be within max_nearest_km
    nearest_dist, _ = tree.query(grid_pts_km, k=1)
    too_far = nearest_dist.reshape(grid_lats.shape) > max_nearest_km
    grid[too_far] = np.nan

    return grid


def smooth_field(grid, sigma_deg=1.5):
    """
    Gaussian smooth with NaN-aware weighting (so coastlines don't bleed).
    sigma_deg is the Gaussian sigma in degrees; converted to grid cells.
    """
    sigma_cells = sigma_deg / GRID_RES
    filled  = np.where(np.isnan(grid), 0.0, grid)
    weight  = np.where(np.isnan(grid), 0.0, 1.0)
    s_data  = gaussian_filter(filled,  sigma=sigma_cells)
    s_weight = gaussian_filter(weight, sigma=sigma_cells)
    with np.errstate(invalid='ignore'):
        result = np.where(s_weight > 0.3, s_data / s_weight, np.nan)
    return result


def adaptive_threshold(smooth_grid, sigma=1.5):
    """Return threshold = sigma * std of the valid (non-NaN) smoothed field."""
    valid = smooth_grid[~np.isnan(smooth_grid)]
    if len(valid) == 0:
        return 0.3
    return float(sigma * valid.std())


def detect_structures(smooth_grid, grid_lats, grid_lons,
                      threshold=None, min_area_km2=15_000,
                      station_lats=None, station_lons=None,
                      min_center_stations=2, center_radius_km=150):
    """
    Find connected positive/negative anomaly regions.

    Returns a list of dicts, one per detected structure:
        center_lat, center_lon, amplitude, sign (+1/-1),
        orientation_deg (0-180, major-axis azimuth from North),
        extent_lat_deg, extent_lon_deg, area_km2

    If station_lats/station_lons are provided, any structure whose center
    has fewer than min_center_stations within center_radius_km is discarded
    (this removes interpolation artefacts over ocean/data-sparse areas).
    """
    if threshold is None:
        threshold = adaptive_threshold(smooth_grid)

    # Build KD-tree for station-center proximity check
    center_tree = None
    if station_lats is not None and station_lons is not None and len(station_lats) > 0:
        center_tree = cKDTree(np.column_stack([
            np.asarray(station_lons) * KM_PER_DEG_LON,
            np.asarray(station_lats) * KM_PER_DEG_LAT,
        ]))

    cell_area_km2 = (GRID_RES * KM_PER_DEG_LAT) * (GRID_RES * KM_PER_DEG_LON)
    min_cells = max(4, int(min_area_km2 / cell_area_km2))

    field = np.nan_to_num(smooth_grid, nan=0.0)
    structures = []

    for sign, mask in [(+1, field > threshold), (-1, field < -threshold)]:
        labeled, n = label(mask)
        for i in range(1, n + 1):
            region = labeled == i
            if region.sum() < min_cells:
                continue

            lat_pts = grid_lats[region]
            lon_pts = grid_lons[region]
            val_pts = field[region]

            center_lat = float(lat_pts.mean())
            center_lon = float(lon_pts.mean())
            amplitude  = float(abs(val_pts).max())

            # PCA to find major axis orientation
            w = np.abs(val_pts)
            w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
            dlat = lat_pts - lat_pts.mean()
            dlon = lon_pts - lon_pts.mean()
            cov = np.array([
                [np.sum(w * dlat**2),       np.sum(w * dlat * dlon)],
                [np.sum(w * dlat * dlon),   np.sum(w * dlon**2)],
            ])
            eigvals, eigvecs = np.linalg.eigh(cov)
            major = eigvecs[:, eigvals.argmax()]       # [dlat, dlon]
            # Convert to compass bearing of the band axis (0-180)
            orientation_deg = float(np.degrees(np.arctan2(major[1], major[0])) % 180)

            # Discard structures over ocean / data-sparse areas
            if center_tree is not None:
                center_pt_km = [center_lon * KM_PER_DEG_LON,
                                center_lat * KM_PER_DEG_LAT]
                n_nearby = len(center_tree.query_ball_point(
                    center_pt_km, r=center_radius_km))
                if n_nearby < min_center_stations:
                    continue

            structures.append({
                'center_lat':     center_lat,
                'center_lon':     center_lon,
                'amplitude':      amplitude,
                'sign':           sign,
                'orientation_deg': orientation_deg,
                'extent_lat_deg': float(lat_pts.std() * 2),
                'extent_lon_deg': float(lon_pts.std() * 2),
                'area_km2':       float(region.sum() * cell_area_km2),
            })

    return structures


def estimate_propagation(grid1, grid2, dt_seconds):
    """
    Estimate the bulk wave-packet propagation vector from two consecutive
    smoothed fields using 2D phase cross-correlation (Kuglin & Hines method).

    Returns dict with speed_kmh, heading_deg (meteorological, from north CW),
    vx_kmh (eastward), vy_kmh (northward).  Returns None if dt <= 0.
    """
    if dt_seconds <= 0:
        return None

    f1 = np.nan_to_num(grid1, nan=0.0)
    f2 = np.nan_to_num(grid2, nan=0.0)

    # Phase correlation
    F1 = np.fft.fft2(f1)
    F2 = np.fft.fft2(f2)
    denom = np.abs(F1) * np.abs(F2)
    denom[denom == 0] = 1.0
    cross = (F2 * np.conj(F1)) / denom
    corr  = np.real(np.fft.ifft2(cross))
    corr  = np.fft.fftshift(corr)

    peak = np.unravel_index(corr.argmax(), corr.shape)
    dy_cells = peak[0] - corr.shape[0] // 2   # positive = northward shift
    dx_cells = peak[1] - corr.shape[1] // 2   # positive = eastward shift

    dy_km = dy_cells * GRID_RES * KM_PER_DEG_LAT
    dx_km = dx_cells * GRID_RES * KM_PER_DEG_LON

    dt_h    = dt_seconds / 3600.0
    vy_kmh  = dy_km / dt_h   # northward
    vx_kmh  = dx_km / dt_h   # eastward
    speed   = np.sqrt(vx_kmh**2 + vy_kmh**2)

    # Meteorological heading: clockwise from North
    heading = float(np.degrees(np.arctan2(vx_kmh, vy_kmh)) % 360)

    # Sanity: discard if propagation > 500 km/h (likely spurious)
    if speed > 500:
        return None

    return {
        'speed_kmh':  float(speed),
        'heading_deg': heading,
        'vx_kmh':     float(vx_kmh),
        'vy_kmh':     float(vy_kmh),
    }


def adaptive_contour_levels(smooth_grid, n_pos=4):
    """
    Return contour levels scaled to the field's own standard deviation.
    Produces n_pos positive and n_pos negative levels.
    """
    valid = smooth_grid[~np.isnan(smooth_grid)]
    if len(valid) == 0:
        return []
    fstd = valid.std()
    # Levels at 0.75, 1.25, 1.75, 2.5 sigma
    mults = [0.75, 1.25, 1.75, 2.5][:n_pos]
    pos   = sorted( m * fstd for m in mults)
    neg   = sorted(-m * fstd for m in mults)
    return neg + pos


def get_contour_paths(smooth_grid, grid_lats, grid_lons, levels=None):
    """
    Generate contour line paths using matplotlib.

    Returns dict mapping float(level) -> list of paths,
    where each path is [[lat, lon], [lat, lon], ...].
    Paths are downsampled to <= 200 points each.
    """
    if levels is None:
        levels = adaptive_contour_levels(smooth_grid)
    if not levels:
        return {}

    field = np.where(np.isnan(smooth_grid), np.nan, smooth_grid)
    result = {}

    try:
        fig, ax = plt.subplots(figsize=(2, 2))
        # contour(X=lons, Y=lats, Z=field)
        cs = ax.contour(grid_lons, grid_lats, field, levels=levels)
        for level_val, segs in zip(cs.levels, cs.allsegs):
            paths = []
            for seg in segs:
                if len(seg) < 2:
                    continue
                # seg shape: (N, 2) with columns [lon, lat]
                pts = np.asarray(seg)
                # Downsample if long
                if len(pts) > 200:
                    idx = np.round(np.linspace(0, len(pts) - 1, 200)).astype(int)
                    pts = pts[idx]
                # Return as [[lat, lon], ...]
                paths.append([[float(p[1]), float(p[0])] for p in pts])
            if paths:
                result[float(level_val)] = paths
        plt.close(fig)
    except Exception:
        plt.close('all')

    return result
