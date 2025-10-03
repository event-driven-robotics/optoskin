import matplotlib.pyplot as plt
import numpy as np

def plot_rate_curves_log(grid, R, med, band, latencies):
    if grid is None: return
    p5, p95 = band
    plt.figure(figsize=(8, 4))
    for rr in R:
        plt.plot(grid, rr, alpha=0.10, lw=1)
    plt.fill_between(grid, p5, p95, alpha=0.30, label="rate 5–95%")
    plt.plot(grid, med, lw=2, label="median rate")
    finite = latencies[np.isfinite(latencies)]
    if finite.size:
        L5, L50, L95 = np.percentile(finite, [5, 50, 95])
        plt.axvline(L50, ls="--", lw=2, label=f"median latency {L50*1e3:.1f} ms")
        plt.axvspan(L5, L95, alpha=0.2, label="latency 5–95%")
    plt.axvline(0, ls=":", lw=1, label="t=0 anchor")
    plt.yscale("log")
    plt.xlabel("Time [s]"); plt.ylabel("Aggregate event rate [Hz] (log)")
    plt.title("Aligned rate rise")
    plt.legend(); plt.tight_layout(); plt.show()

def plot_latency_hist(latencies):
    finite = latencies[np.isfinite(latencies)]
    if finite.size == 0: return
    plt.figure(figsize=(6, 4))
    plt.hist(finite * 1e3, bins=20)
    plt.xlabel("Latency [ms]"); plt.ylabel("Count")
    plt.title("Latency distribution")
    plt.tight_layout(); plt.show()

def plot_roc(fpr, tpr, hs, best_idx=None):
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, marker='o', lw=1)
    if best_idx is not None and np.isfinite(fpr[best_idx]) and np.isfinite(tpr[best_idx]):
        plt.scatter([fpr[best_idx]], [tpr[best_idx]], s=90, edgecolor='k', zorder=5)
        plt.text(fpr[best_idx], tpr[best_idx], f"h={hs[best_idx]:.1f}", fontsize=9,
                 ha='left', va='bottom')
    # annotate a few points
    for i in range(0, len(hs), max(1, len(hs)//6)):
        if np.isfinite(fpr[i]) and np.isfinite(tpr[i]):
            plt.text(fpr[i], tpr[i], f"{hs[i]:.1f}", fontsize=8)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC (CUSUM threshold sweep)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_side_by_side(grid, R, med, band, latencies, y_floor=1e-1, ax=None, colour='b', label=''):
    """
    Single-plot version: shows only the aligned rate rise (log-y), full-width.
    - Zero (or negative) rates are clipped to `y_floor` so they appear on a log axis.
    - Y-axis ticks start with 0, then …, then log-decade ticks above `y_floor`.
    - Time axis is shifted so that the detected median latency is at x=0.
    - Keeps the dashed vertical line for the median (no label), removes t=0 anchor and latency 5–95% span.
    """
    if grid is None:
        return
    import numpy as np
    import matplotlib.pyplot as plt

    p5, p95 = band
    finite_lat = latencies[np.isfinite(latencies)]
    L50 = float(np.median(finite_lat)) if finite_lat.size else 0.0

    # shift times so median latency is at zero
    grid_shift = grid - L50

    # --- clip zeros (and negatives) to y_floor for plotting on log scale ---
    def clip(arr):
        return np.where(arr > y_floor, arr, y_floor)

    R_plot   = np.array([clip(rr) for rr in R])
    med_plot = clip(med)
    p5_plot  = clip(p5)
    p95_plot = clip(p95)

    # --- build custom y-ticks: [0, …, decades above y_floor] ---
    ymax = float(np.nanmax([np.nanmax(R_plot), np.nanmax(p95_plot), np.nanmax(med_plot)]))
    if not np.isfinite(ymax) or ymax <= y_floor:
        ymax = y_floor * 10.0

    # determine log-decade ticks strictly above y_floor
    exp_floor = int(np.floor(np.log10(y_floor)))
    exp_hi    = int(np.ceil(np.log10(ymax)))
    decades = [10.0**e for e in range(exp_floor + 1, exp_hi + 1)]

    # an intermediate tick to host the "…" label
    ellipsis_tick = y_floor * 3.0
    ticks = [y_floor, ellipsis_tick] + decades

    def decade_label(val):
        e = int(round(np.log10(val)))
        return "1" if e == 0 else f"1e{e}"

    tick_labels = ["0", "…"] + [decade_label(v) for v in decades]

    # --- plotting ---
    if ax is None:
        plt.figure(figsize=(10.5, 4.2))
    # individual trials (light)
    #for rr in R_plot:
    #    plt.plot(grid_shift, rr, alpha=0.10, lw=1)
    # rate 5–95% band + median rate
    plt.plot(grid_shift, med_plot, colour, lw=2, label=label+" median instantaneous rate")
    plt.fill_between(grid_shift, p5_plot, p95_plot, color=colour, alpha=0.30, label=label+" inst. rate 5–95%")

    # dashed median line at x=0 (no label)
    #plt.axvline(0.0, ls="--", lw=2)

    #plt.set_yscale("log")
    plt.yscale("log")
    plt.ylim(y_floor, ymax * 1.1)
    plt.yticks(ticks, tick_labels)

    plt.xlabel("Time from median latency [s]")
    plt.ylabel("Event rate [Hz] (log, zeros shown as 0)")
    #plt.title("Aligned rate rise (time shifted so median latency = 0)")
    plt.legend()
    plt.tight_layout()
    plt.show()
