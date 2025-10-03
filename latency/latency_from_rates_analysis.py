# latency_from_rates_roc_presets.py
import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

# -------------------- Parameters & Presets --------------------

@dataclass
class DetectionParams:
    dt: float = 2e-4             # 0.2 ms bins
    smooth_sigma: float = 5e-4    # 0.5 ms smoothing
    alpha: float = 4.0            # assumed step-up factor (λ1 = α·λ0)
    hysteresis_bins: int = 3      # consecutive bins to confirm change

def preset_balanced() -> DetectionParams:
    """Good starting point for ~10 ms latencies, onset ~250 ms."""
    return DetectionParams(dt=2e-4, smooth_sigma=5e-4, alpha=4.0, hysteresis_bins=3)

def preset_low_false_alarm() -> DetectionParams:
    """Lower FPR; slightly slower trigger."""
    return DetectionParams(dt=2e-4, smooth_sigma=7e-4, alpha=5.0, hysteresis_bins=4)

# -------------------- Core helpers --------------------

def bin_counts(times: np.ndarray, t0: float, t1: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.arange(t0, t1 + dt, dt)
    counts, _ = np.histogram(times, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts.astype(float), centers

def gaussian_kernel(sigma: float, dt: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0])
    hw = max(1, int(round(4 * sigma / dt)))
    t = np.arange(-hw, hw + 1) * dt
    k = np.exp(-0.5 * (t / sigma) ** 2)
    k /= k.sum()
    return k

def smooth_counts(counts: np.ndarray, dt: float, sigma: float) -> np.ndarray:
    k = gaussian_kernel(sigma, dt)
    return np.convolve(counts, k, mode="same")

def poisson_cusum_first_cross(
    counts: np.ndarray, dt: float, lam0: float, alpha: float, h: float, hysteresis_bins: int
) -> Optional[int]:
    """One-sided Poisson CUSUM for step-up from λ0 to λ1=α·λ0. Returns index or None."""
    lam1 = max(1e-12, alpha * max(lam0, 1e-12))
    log_r = math.log(lam1 / max(lam0, 1e-12))
    delta = (lam1 - lam0) * dt
    S, consec = 0.0, 0
    for i, x in enumerate(counts):
        L = x * log_r - delta
        S = max(0.0, S + L)
        if S >= h:
            consec += 1
            if consec >= hysteresis_bins:
                return i - hysteresis_bins + 1
        else:
            consec = 0
    return None

# --- helper to create reduced data condition

def random_reduce(arr: np.ndarray, reduction_factor: int, rng=None) -> np.ndarray:
    """
    Randomly reduce the size of a 1D numpy array by a given factor.

    Parameters
    ----------
    arr : np.ndarray
        1D array of floats (or any dtype).
    reduction_factor : int
        Factor by which to reduce. The output length will be
        round(len(arr) / reduction_factor).
    rng : np.random.Generator or None
        Optional numpy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Reduced array with elements randomly chosen from arr.
    """
    if reduction_factor <= 0:
        raise ValueError("reduction_factor must be positive")

    n_out = int(round(len(arr) / reduction_factor))
    if n_out <= 0:
        return np.array([], dtype=arr.dtype)

    if rng is None:
        rng = np.random.default_rng()

    idx = rng.choice(len(arr), size=n_out, replace=False)
    return arr[np.sort(idx)]


# -------------------- Detection & Metrics --------------------

def estimate_lambda0_from_bg(bg: np.ndarray) -> float:
    if bg is None or len(bg) < 2:
        return 1e-3
    span = max(bg.max() - bg.min(), 1e-9)
    return len(bg) / span

def detect_latencies(
    trial_event_times: List[np.ndarray], bg_event_times: List[np.ndarray],
    h: float, params: DetectionParams
) -> np.ndarray:
    latencies = np.full(len(trial_event_times), np.nan, dtype=float)
    for i, ts in enumerate(trial_event_times):
        ts = np.asarray(ts)
        if ts.size == 0:
            continue
        lam0 = estimate_lambda0_from_bg(np.asarray(bg_event_times[i]))
        counts, centers = bin_counts(ts, 0.0, ts.max(), params.dt)
        smoothed = smooth_counts(counts, params.dt, params.smooth_sigma)
        idx = poisson_cusum_first_cross(smoothed, params.dt, lam0, params.alpha, h, params.hysteresis_bins)
        if idx is not None:
            latencies[i] = centers[idx]
    return latencies

def _cusum_count_triggers(counts: np.ndarray, dt: float, lam0: float,
                          alpha: float, h: float, hysteresis_bins: int,
                          cooldown_time: float = 0.0) -> int:
    """
    Count how many times the one-sided Poisson CUSUM crosses the threshold.
    After each confirmed crossing, pause accumulation for `cooldown_time` seconds
    (i.e., ignore evidence and keep the statistic reset during that interval).

    Parameters
    ----------
    counts : np.ndarray
        Counts per bin (after any smoothing).
    dt : float
        Bin width in seconds.
    lam0 : float
        Baseline rate [Hz].
    alpha : float
        Post-change multiplier (λ1 = α * λ0).
    h : float
        CUSUM threshold.
    hysteresis_bins : int
        Number of consecutive bins with S >= h required to confirm a trigger.
    cooldown_time : float, optional
        Time (seconds) to pause after each trigger. Default 0.0 s.
        Effective bins skipped = round(cooldown_time / dt).

    Returns
    -------
    int
        Number of triggers detected.
    """
    import math

    lam1 = max(1e-12, alpha * max(lam0, 1e-12))
    log_r = math.log(lam1 / max(lam0, 1e-12))
    delta = (lam1 - lam0) * dt

    cooldown_bins = int(round(max(0.0, cooldown_time) / dt))
    cool = 0  # countdown bins remaining in cooldown

    triggers = 0
    S, consec = 0.0, 0

    for x in counts:
        if cool > 0:
            # During cooldown: ignore evidence and keep stat reset
            cool -= 1
            S = 0.0
            consec = 0
            continue

        L = x * log_r - delta
        S = max(0.0, S + L)

        if S >= h:
            consec += 1
            if consec >= hysteresis_bins:
                triggers += 1
                # Reset and start cooldown
                S = 0.0
                consec = 0
                cool = cooldown_bins
        else:
            consec = 0

    return triggers


def false_alarm_rate_per_second(bg_event_times: List[np.ndarray], h: float, params: DetectionParams, cooldown_time: float) -> float:
    """
    Returns the false-alarm rate in triggers per second over all background snippets.
    """
    total_triggers = 0
    total_seconds = 0.0

    for bg in bg_event_times:
        bg = np.asarray(bg)
        if bg.size < 2:
            continue
        span = float(bg.max() - bg.min())
        if span <= 0:
            continue

        lam0 = estimate_lambda0_from_bg(bg)
        counts, centers = bin_counts(bg, 0.0, bg.max(), params.dt)
        smoothed = smooth_counts(counts, params.dt, params.smooth_sigma)
        num_triggers = _cusum_count_triggers(
            smoothed, params.dt, lam0, params.alpha, h, params.hysteresis_bins,
            cooldown_time
        )
        total_triggers += num_triggers
        total_seconds += span

    return (total_triggers / total_seconds) if total_seconds > 0 else np.nan


def false_alarm_rate(bg_event_times: List[np.ndarray], h: float, params: DetectionParams) -> float:
    fa, den = 0, 0
    for bg in bg_event_times:
        bg = np.asarray(bg)
        if bg.size < 2:
            continue
        lam0 = estimate_lambda0_from_bg(bg)
        counts, centers = bin_counts(bg, 0.0, bg.max(), params.dt)
        smoothed = smooth_counts(counts, params.dt, params.smooth_sigma)
        idx = poisson_cusum_first_cross(smoothed, params.dt, lam0, params.alpha, h, params.hysteresis_bins)
        den += 1
        if idx is not None:
            fa += 1
    return fa / den if den > 0 else np.nan

def hits_within_median_window(latencies: np.ndarray, window: float):
    """
    Returns (hits_mask, L50), where hits_mask is True for detections within ±window
    of the median detected latency (L50). If no finite detections, returns all False and L50=np.nan.
    """
    lat = np.asarray(latencies, dtype=float)
    finite = np.isfinite(lat)
    if not np.any(finite):
        return np.zeros_like(lat, dtype=bool), np.nan
    L50 = float(np.median(lat[finite]))
    hits_mask = finite & (np.abs(lat - L50) <= window)
    L5 = float(np.percentile(lat[hits_mask], 5))
    L95 = float(np.percentile(lat[hits_mask], 95))
    return hits_mask, L5, L50, L95


def compute_roc(
    trial_event_times: List[np.ndarray], bg_event_times: List[np.ndarray],
    params: DetectionParams, h_values: np.ndarray, latency_max: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tpr_list, fpr_list, l5_list , l50_list , l95_list = [], [], [], [], []
    n_trials = len(trial_event_times)
    for h in h_values:
        lat = detect_latencies(trial_event_times, bg_event_times, h, params)
        hits_mask, L5, L50, L95 = hits_within_median_window(lat, latency_max)
        tpr = hits_mask.sum() / n_trials if n_trials > 0 else np.nan
        #fpr = false_alarm_rate(bg_event_times, h, params)
        fpr = false_alarm_rate_per_second(bg_event_times, h, params, L95-L5)
        tpr_list.append(tpr); fpr_list.append(fpr)
        l5_list.append(L5)
        l50_list.append(L50)
        l95_list.append(L95)
    return np.array(fpr_list), np.array(tpr_list),  h_values, np.array(l5_list), np.array(l50_list), np.array(l95_list)

def select_h_by_constraint(
    trial_event_times: List[np.ndarray], bg_event_times: List[np.ndarray],
    params: DetectionParams, h_values: np.ndarray, latency_max: float,
    target_tpr: float = 0.95
) -> Tuple[float, dict]:
    """Pick h with TPR≥target_tpr (within latency_max). Among feasible points, minimize FPR;
       if none feasible, pick the one with largest TPR (break ties by lower FPR)."""
    fpr, tpr, hs, l5s , l50s , l95s = compute_roc(trial_event_times, bg_event_times, params, h_values, latency_max)

    feasible = tpr >= target_tpr
    if np.any(feasible):
        cand = np.where(feasible)[0]
        best_idx = cand[np.nanargmin(fpr[cand])]
    else:
        # closest to target: max TPR; tie-breaker: min FPR
        max_tpr = np.nanmax(tpr)
        cand = np.where(tpr == max_tpr)[0]
        best_idx = cand[np.nanargmin(fpr[cand])]
    h_star = float(hs[best_idx])
    return h_star, {"fpr": fpr, "tpr": tpr, "hs": hs, "best_idx": int(best_idx)}

# -------------------- Rate curve aggregation (for plots) --------------------


def build_rate_curves(trial_event_times, params, t_min=None, t_max=None):
    """
    Build aligned rate curves on a common grid, with edge-robust smoothing.
    - Common grid: [t_min, t_max] with step params.dt; if not provided, we use
      t_min = max(trial.min()) and t_max = min(trial.max()) to keep only
      the time region present in *all* trials (avoids interpolation artifacts).
    - Smoothing uses reflect padding to remove start/end bias.

    Returns
      grid : (T,) time centers
      R    : (N, T) rate per trial (Hz) on common grid
      med  : (T,) median rate across trials
      (p5, p95) : 5th and 95th percentile envelopes across trials
    """
    import numpy as np

    # collect valid trials
    trials = [np.asarray(ts) for ts in trial_event_times if np.asarray(ts).size > 0]
    if not trials:
        return None, None, None, None

    # choose common window if not supplied
    if t_min is None:
        t_min = max(ts.min() for ts in trials)
    if t_max is None:
        t_max = min(ts.max() for ts in trials)

    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return None, None, None, None

    dt = params.dt
    edges = np.arange(t_min, t_max + dt, dt)  # bin edges
    grid  = 0.5 * (edges[:-1] + edges[1:])   # bin centers

    # prebuild Gaussian kernel
    def gaussian_kernel(sigma, dt):
        if sigma <= 0:
            return np.array([1.0])
        hw = max(1, int(round(4 * sigma / dt)))
        t  = np.arange(-hw, hw + 1) * dt
        k  = np.exp(-0.5 * (t / sigma) ** 2)
        k /= k.sum()
        return k, hw

    k, hw = gaussian_kernel(params.smooth_sigma, dt)

    R = []
    for ts in trials:
        # bin on the common grid
        counts, _ = np.histogram(ts, bins=edges)

        # smooth with *reflect* padding to avoid edge artifacts
        if len(k) > 1:
            pad = hw  # pad width in bins
            padded = np.pad(counts.astype(float), (pad, pad), mode='reflect')
            smoothed = np.convolve(padded, k, mode='same')[pad:-pad]
        else:
            smoothed = counts.astype(float)

        # convert to Hz
        rate_hz = smoothed / dt
        R.append(rate_hz)

    R = np.asarray(R)  # (N, T)

    # robust summaries across trials
    med = np.nanmedian(R, axis=0)
    p5  = np.nanpercentile(R, 5, axis=0)
    p95 = np.nanpercentile(R, 95, axis=0)

    return grid, R, med, (p5, p95)


# -------------------- Plotting --------------------


# -------------------- Top-level runner --------------------

def run_pipeline(
    trial_event_times: List[np.ndarray],
    bg_event_times: List[np.ndarray],
    params: DetectionParams = preset_balanced(),
    h_values: np.ndarray = np.linspace(5, 20, 16),   # sweep 5..20
    latency_max: float = 0.020,                      # 20 ms TPR window
    target_tpr: float = 0.95
):
    # 1) choose h by constraint
    h_star, roc = select_h_by_constraint(trial_event_times, bg_event_times, params,
                                         h_values, latency_max, target_tpr)
    fpr_vec, tpr_vec, hs, best_idx = roc["fpr"], roc["tpr"], roc["hs"], roc["best_idx"]

    # 2) detect latencies at chosen h
    latencies = detect_latencies(trial_event_times, bg_event_times, h_star, params)
    finite = latencies[np.isfinite(latencies)]
    L50 = np.median(finite) if finite.size else np.nan
    hits = finite[np.abs(finite - L50) < latency_max]
    L5, L95 = (np.percentile(hits, [5, 95]) if hits.size else (np.nan, np.nan))
    latency_distribution = L95 - L5
    #fpr_eff = false_alarm_rate(bg_event_times, h_star, params)
    fpr_s_eff = false_alarm_rate_per_second(bg_event_times, h_star, params, latency_distribution)

    # 3) build rate curves for plots
    grid, R, med, band = build_rate_curves(trial_event_times, params)

    # 4) print summary
    print("\n=== Detection summary ===")
    print(f"Chosen threshold h*: {h_star:.2f}  (target TPR ≥ {target_tpr:.2f}, window {latency_max*1e3:.0f} ms)")
    print(f"Trials detected within window: {(np.isfinite(latencies) & (np.abs(latencies -L50) < latency_max)).sum()} / {len(trial_event_times)}")
    print(f"Median latency: {L50*1e3:.2f} ms")
    print(f"Latency 5–95%: [{L5*1e3:.2f}, {L95*1e3:.2f}] ms")
    print(f"Latency distribution: {latency_distribution*1e3:.2f} ms")
    #print(f"Effective FPR at h*: {fpr_eff*100:.2f}%")
    print(f"Effective FPR per second at h*: {fpr_s_eff:.2f}")

    # 5) plots
    #from latency_from_rates_plots import plot_side_by_side
    
    #plot_rate_curves_log(grid, R, med, band, latencies)
    #plot_latency_hist(latencies)
    #plot_roc(fpr_vec, tpr_vec, hs, best_idx=best_idx)
    #plot_side_by_side(grid, R, med, band, latencies)

    #grid, R, med, band = build_rate_curves(bg_event_times, params)
    #plot_side_by_side(grid, R, med, band, latencies, y_floor=1e-1, ax='already there', colour='r')

    return {
        "h_star": h_star,
        "latencies": latencies,
        "median_latency": L50,
        "latency_5_95": (L5, L95),
        "fpr_s_at_h_star": fpr_s_eff,
        "roc": {"fpr": fpr_vec, "tpr": tpr_vec, "hs": hs, "best_idx": best_idx},
        "grid": grid, "R": R, "rate_median": med, "rate_band": band
    }

