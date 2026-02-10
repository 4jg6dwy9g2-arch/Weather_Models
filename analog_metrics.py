"""
Analog pattern-matching metrics for ERA5 historical analog search.

Provides multi-metric composite scoring that captures:
  1. Area-weighted pattern correlation (latitude-weighted Pearson)
  2. Spatial gradient correlation (ridge/trough position similarity)
  3. RMSE-based amplitude similarity (penalises amplitude differences)
  4. EOF projection similarity (optional, requires pre-built cache)

Usage
-----
    from analog_metrics import (
        build_lat_weights,
        weighted_pearson,
        gradient_correlation,
        rmse_similarity,
        compute_composite_score,
        build_eof_cache,
        load_eof_cache,
        eof_similarity,
    )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Path for the persistent EOF cache on the external drive
_EOF_CACHE_PATH = Path("/Volumes/T7/Weather_Models/era5/analog_eof_cache.npz")

# ---------------------------------------------------------------------------
# Latitude area-weighting helpers
# ---------------------------------------------------------------------------

def build_lat_weights(lats: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return a 2-D cosine(lat) weight matrix matching *shape* (n_lat, n_lon).

    Parameters
    ----------
    lats:
        1-D array of latitudes in degrees.
    shape:
        (n_lat, n_lon) target shape – must match ``len(lats)``.

    Returns
    -------
    np.ndarray of shape (n_lat, n_lon), values in [0, 1].
    """
    n_lat, n_lon = shape
    assert len(lats) == n_lat, f"len(lats)={len(lats)} != n_lat={n_lat}"
    w = np.cos(np.deg2rad(lats))
    w = np.clip(w, 0.0, None)
    w_2d = np.tile(w[:, np.newaxis], (1, n_lon))
    # Normalise so that weights sum to 1 (for flat arrays)
    total = w_2d.sum()
    if total > 0:
        w_2d = w_2d / total
    return w_2d


# ---------------------------------------------------------------------------
# Individual metrics (all operate on 1-D flattened arrays or 2-D grids)
# ---------------------------------------------------------------------------

def weighted_pearson(
    a1_flat: np.ndarray,
    a2_flat: np.ndarray,
    weights_flat: np.ndarray,
) -> float:
    """Area-weighted Pearson correlation between two flattened anomaly arrays.

    Parameters
    ----------
    a1_flat, a2_flat:
        1-D arrays of the same length (grid-point anomalies).
    weights_flat:
        1-D array of non-negative weights (need not sum to 1).

    Returns
    -------
    float in [-1, 1], or NaN if calculation is impossible.
    """
    finite = np.isfinite(a1_flat) & np.isfinite(a2_flat) & np.isfinite(weights_flat)
    if finite.sum() < 10:
        return float("nan")

    a1 = a1_flat[finite]
    a2 = a2_flat[finite]
    w  = weights_flat[finite]

    w_sum = w.sum()
    if w_sum <= 0:
        return float("nan")

    mu1 = (w * a1).sum() / w_sum
    mu2 = (w * a2).sum() / w_sum

    d1 = a1 - mu1
    d2 = a2 - mu2

    cov   = (w * d1 * d2).sum() / w_sum
    var1  = (w * d1 ** 2).sum() / w_sum
    var2  = (w * d2 ** 2).sum() / w_sum

    denom = np.sqrt(var1 * var2)
    if denom < 1e-12:
        return float("nan")

    return float(np.clip(cov / denom, -1.0, 1.0))


def gradient_correlation(a1: np.ndarray, a2: np.ndarray) -> float:
    """Pearson correlation of the spatial gradient fields.

    Captures ridge/trough *positions* better than raw anomaly correlation
    because two patterns with the same ridges shifted by ~2° will score
    lower here than they would with plain Pearson.

    Parameters
    ----------
    a1, a2:
        2-D anomaly arrays of shape (n_lat, n_lon). Missing data as NaN.

    Returns
    -------
    float in [-1, 1], or NaN.
    """
    if a1.shape != a2.shape:
        return float("nan")

    # Compute gradient magnitude at each grid point
    def _grad_mag(arr: np.ndarray) -> np.ndarray:
        dy, dx = np.gradient(arr)
        return np.hypot(dy, dx)

    g1 = _grad_mag(a1).ravel()
    g2 = _grad_mag(a2).ravel()

    finite = np.isfinite(g1) & np.isfinite(g2)
    if finite.sum() < 10:
        return float("nan")

    g1f, g2f = g1[finite], g2[finite]
    std1, std2 = g1f.std(), g2f.std()
    if std1 < 1e-12 or std2 < 1e-12:
        return float("nan")

    r = float(np.corrcoef(g1f, g2f)[0, 1])
    return float(np.clip(r, -1.0, 1.0))


def rmse_similarity(
    a1_flat: np.ndarray,
    a2_flat: np.ndarray,
) -> float:
    """Amplitude-aware similarity: ``1 / (1 + RMSE / RMS_current)``.

    Unlike Pearson, this returns a *lower* score when one pattern is a
    weaker or stronger version of the other.

    Returns
    -------
    float in (0, 1].  1.0 means identical patterns.
    """
    finite = np.isfinite(a1_flat) & np.isfinite(a2_flat)
    if finite.sum() < 10:
        return 0.0

    a1f = a1_flat[finite]
    a2f = a2_flat[finite]

    rmse = float(np.sqrt(np.mean((a1f - a2f) ** 2)))
    rms_current = float(np.sqrt(np.mean(a1f ** 2)))

    if rms_current < 1e-6:
        return 0.0

    return float(1.0 / (1.0 + rmse / rms_current))


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def compute_composite_score(
    lat_pearson: float,
    grad_corr: float,
    rmse_sim: float,
    eof_sim: Optional[float] = None,
) -> float:
    """Weighted combination of individual metrics.

    Weights
    -------
    With EOF similarity available:
        0.40 × lat_pearson + 0.35 × grad_corr + 0.15 × rmse_sim + 0.10 × eof_sim

    Without EOF:
        0.45 × lat_pearson + 0.40 × grad_corr + 0.15 × rmse_sim

    Any NaN component is treated as 0.0 and its weight is redistributed
    proportionally among finite components.

    Returns
    -------
    float in [0, 1] (scores that are negative are clamped to 0).
    """
    if eof_sim is not None and np.isfinite(eof_sim):
        components = [
            (lat_pearson, 0.40),
            (grad_corr,   0.35),
            (rmse_sim,    0.15),
            (eof_sim,     0.10),
        ]
    else:
        components = [
            (lat_pearson, 0.45),
            (grad_corr,   0.40),
            (rmse_sim,    0.15),
        ]

    # Map correlations from [-1,1] to [0,1] for Pearson/grad scores
    def _to_score(val: float) -> float:
        if not np.isfinite(val):
            return float("nan")
        return (val + 1.0) / 2.0  # [-1,1] → [0,1]

    weighted_sum = 0.0
    weight_total = 0.0
    for val, w in components:
        if val is rmse_sim or val is eof_sim:
            # rmse_sim and eof_sim are already in [0,1]
            converted = val if np.isfinite(val) else float("nan")
        else:
            converted = _to_score(val)

        if np.isfinite(converted):
            weighted_sum += converted * w
            weight_total += w

    if weight_total <= 0:
        return 0.0

    # Renormalise to handle any missing components
    score = weighted_sum / weight_total
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# EOF cache  (optional – requires scikit-learn)
# ---------------------------------------------------------------------------

try:
    from sklearn.decomposition import PCA as _PCA
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    logger.warning("scikit-learn not found; EOF-based similarity will be unavailable")


def build_eof_cache(
    era5_ds,            # xarray.Dataset with 'z' variable and 'time' dimension
    climatology: dict,  # {doy: np.ndarray} from the analog endpoints
    pressure_coord: str = "pressure_level",
    lat_coord: str = "latitude",
    lon_coord: str = "longitude",
    n_components: int = 20,
    clim_years: tuple[int, int] = (1990, 2021),
    cache_path: Path = _EOF_CACHE_PATH,
) -> dict:
    """Fit a PCA on normalised ERA5 z500 anomalies and save results to disk.

    Parameters
    ----------
    era5_ds:
        Combined xarray.Dataset (all NH ERA5 files concatenated).
    climatology:
        Day-of-year → mean z500 grid dict (same structure as the endpoints build).
    n_components:
        Number of EOFs to retain (default 20 captures ~85-90% variance).
    cache_path:
        Path to save the .npz cache file.

    Returns
    -------
    dict with keys: ``components``, ``mean``, ``times``, ``scores``, ``shape``.
    """
    import pandas as pd

    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required for EOF cache building")

    logger.info("Building EOF cache from ERA5 z500 data …")

    # Build matrix of anomaly vectors
    import xarray as xr

    times = era5_ds.time.values
    anoms = []
    valid_times = []

    for t in times:
        try:
            t_pd = pd.Timestamp(t)
            if not (clim_years[0] <= t_pd.year <= clim_years[1]):
                continue

            doy = t_pd.dayofyear
            z = era5_ds["z"].sel(time=t, **{pressure_coord: 500}).values

            if doy in climatology:
                anom = z - climatology[doy]
            else:
                anom = z - np.nanmean(z, axis=1, keepdims=True)

            if np.isfinite(anom).mean() < 0.9:
                continue

            anoms.append(anom.ravel())
            valid_times.append(t)
        except Exception:
            continue

    if len(anoms) < n_components + 1:
        raise ValueError(f"Not enough valid timesteps ({len(anoms)}) for PCA with n_components={n_components}")

    X = np.array(anoms)  # (n_times, n_grid)

    pca = _PCA(n_components=n_components)
    scores = pca.fit_transform(X)  # (n_times, n_components)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(cache_path),
        components=pca.components_,
        mean=pca.mean_,
        scores=scores,
        times=np.array([str(t) for t in valid_times]),
        explained_variance_ratio=pca.explained_variance_ratio_,
        grid_shape=np.array(list(anoms[0].shape)),  # flattened length
    )

    logger.info(
        f"EOF cache saved to {cache_path}: {n_components} EOFs, "
        f"{scores.shape[0]} timesteps, "
        f"explained variance = {pca.explained_variance_ratio_.sum():.1%}"
    )

    return {
        "components": pca.components_,
        "mean": pca.mean_,
        "scores": scores,
        "times": valid_times,
        "shape": anoms[0].shape,
    }


def load_eof_cache(
    cache_path: Path = _EOF_CACHE_PATH,
    era5_files: Optional[list] = None,
    force_rebuild: bool = False,
) -> Optional[dict]:
    """Load the EOF cache from disk.

    Treats the cache as stale if any ERA5 file is newer than the cache file.

    Returns
    -------
    dict with EOF data or ``None`` if the cache doesn't exist / is stale.
    """
    if not cache_path.exists():
        logger.info("EOF cache not found at %s", cache_path)
        return None

    if not force_rebuild and era5_files:
        cache_mtime = cache_path.stat().st_mtime
        for f in era5_files:
            if Path(f).stat().st_mtime > cache_mtime:
                logger.info("ERA5 files are newer than EOF cache – cache is stale")
                return None

    try:
        data = np.load(str(cache_path), allow_pickle=False)
        logger.info(
            "Loaded EOF cache: %d components, %d timesteps",
            data["components"].shape[0],
            len(data["times"]),
        )
        return {
            "components": data["components"],
            "mean": data["mean"],
            "scores": data["scores"],
            "times": data["times"],
            "explained_variance_ratio": data["explained_variance_ratio"],
        }
    except Exception as e:
        logger.warning("Failed to load EOF cache: %s", e)
        return None


def eof_similarity(
    current_anomaly: np.ndarray,
    eof_cache: dict,
    date_idx: int,
) -> float:
    """Compute similarity between current pattern and a historical analog in PC space.

    Parameters
    ----------
    current_anomaly:
        2-D or 1-D current z500 anomaly array (must match EOF grid size).
    eof_cache:
        Dict returned by :func:`load_eof_cache`.
    date_idx:
        Index into ``eof_cache['scores']`` for the historical date.

    Returns
    -------
    float in [0, 1] (1 = identical in PC space, approaches 0 as distance grows).
    """
    try:
        flat = current_anomaly.ravel()
        finite = np.isfinite(flat)
        if finite.mean() < 0.9:
            return float("nan")

        components = eof_cache["components"]   # (n_eof, n_grid)
        mean_vec   = eof_cache["mean"]          # (n_grid,)
        hist_score = eof_cache["scores"][date_idx]  # (n_eof,)

        # Project current anomaly into PC space
        centered = flat - mean_vec
        current_score = components @ centered   # (n_eof,)

        # Euclidean distance in PC space (normalise by number of components)
        diff = current_score - hist_score
        dist = float(np.sqrt(np.mean(diff ** 2)))

        # Convert to [0,1] similarity; scale by a typical distance (~50 m)
        typical_dist = 50.0
        sim = float(np.exp(-dist / typical_dist))
        return float(np.clip(sim, 0.0, 1.0))

    except Exception as e:
        logger.debug("eof_similarity error: %s", e)
        return float("nan")
