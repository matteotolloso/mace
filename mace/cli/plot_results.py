"""
Notebook-friendly UQ parity plotting with alpha^2 calibration on VALIDATION and evaluation on TEST.

What you get:
- Load (y_true, y_pred, var_pred) from extxyz for val + test
- Fit a global variance scale alpha^2 on validation only
- Apply it to test variances
- Compute ENCE + log-log slope/corr on test
- Plot parity RMV vs RMSE on test (binned), optionally show pre/post-calibration curves

Designed to be pasted into a Jupyter notebook cell (no CLI, no __main__ needed).
"""

from typing import Optional, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

from ase.io import read as ase_read


# -----------------------
# Core utilities
# -----------------------

def _as_1d(x):
    x = np.asarray(x)
    return x.reshape(-1)


def load_uq_from_extxyz(
    path: str,
    *,
    ref_energy_key: str,
    pred_energy_key: str,
    pred_var_key: Optional[str] = None,
    pred_logvar_key: Optional[str] = None,
    pred_std_key: Optional[str] = None,
    per_atom: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load y_true, y_pred, var_pred, natoms from an extxyz written by eval_configs.
    Provide exactly one among (pred_var_key, pred_logvar_key, pred_std_key).

    If per_atom=True:
      E_pa = E / N
      Var_pa = Var / N^2
    """
    db = ase_read(path, ":")
    y_true = np.array([float(at.info[ref_energy_key]) for at in db], dtype=float)
    y_pred = np.array([float(at.info[pred_energy_key]) for at in db], dtype=float)
    nat = np.array([len(at) for at in db], dtype=float)

    if pred_var_key is not None:
        var = np.array([float(at.info[pred_var_key]) for at in db], dtype=float)
    elif pred_logvar_key is not None:
        logvar = np.array([float(at.info[pred_logvar_key]) for at in db], dtype=float)
        var = np.exp(logvar)
    elif pred_std_key is not None:
        std = np.array([float(at.info[pred_std_key]) for at in db], dtype=float)
        var = std**2
    else:
        raise ValueError("Provide one of pred_var_key, pred_logvar_key, pred_std_key")

    var = np.maximum(var, 0.0)

    if per_atom:
        y_true = y_true / nat
        y_pred = y_pred / nat
        var = var / (nat**2)

    return y_true, y_pred, var, nat


def fit_alpha2_on_validation(
    y_true_val: np.ndarray,
    y_pred_val: np.ndarray,
    var_pred_val: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """
    Fit scalar alpha^2 on validation:
      minimize || r^2 - s * v ||_2
      s = (v^T r2)/(v^T v)
    """
    y_true_val = _as_1d(y_true_val)
    y_pred_val = _as_1d(y_pred_val)
    v = np.maximum(_as_1d(var_pred_val), eps)
    r2 = (y_true_val - y_pred_val) ** 2
    s = float(np.dot(v, r2) / np.dot(v, v))
    return max(s, 0.0)


def quantile_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile edges for robust binning."""
    values = _as_1d(values)
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, q)
    edges = np.unique(edges)
    if len(edges) < 3:
        raise ValueError(
            "Not enough unique quantile edges (variance values may be constant). "
            "Try fewer bins or disable binning."
        )
    return edges


def make_bins(values: np.ndarray, *, n_bins: int, min_count: int) -> Tuple[list, np.ndarray]:
    """
    Bin indices by quantiles of `values`. Returns (bins, counts).
    """
    edges = quantile_bin_edges(values, n_bins)
    values = _as_1d(values)

    bins = []
    counts = []
    for j in range(len(edges) - 1):
        lo, hi = edges[j], edges[j + 1]
        if j == len(edges) - 2:
            m = (values >= lo) & (values <= hi)
        else:
            m = (values >= lo) & (values < hi)
        idx = np.where(m)[0]
        if idx.size >= min_count:
            bins.append(idx)
            counts.append(idx.size)

    if len(bins) == 0:
        raise ValueError("No bins survive min_count. Decrease min_count or n_bins.")
    return bins, np.array(counts, dtype=int)


def rmv_rmse_per_bin(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    var_pred: np.ndarray,
    bins: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RMV(b)  = sqrt(mean(var_pred in bin))
    RMSE(b) = sqrt(mean((y-yhat)^2 in bin))
    """
    y_true = _as_1d(y_true)
    y_pred = _as_1d(y_pred)
    var_pred = _as_1d(var_pred)

    rmv = []
    rmse = []
    for idx in bins:
        rmv.append(np.sqrt(np.mean(var_pred[idx])))
        rmse.append(np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2)))
    return np.array(rmv), np.array(rmse)


def ence(rmv: np.ndarray, rmse: np.ndarray, *, eps: float = 1e-12) -> float:
    rmv = np.maximum(_as_1d(rmv), eps)
    rmse = _as_1d(rmse)
    return float(np.mean(np.abs(rmv - rmse) / rmv))


def loglog_fit_stats(x: np.ndarray, y: np.ndarray, *, eps: float = 1e-12) -> Dict[str, float]:
    """
    Fit log(y) = a log(x) + b; return slope, intercept, corr.
    """
    x = np.maximum(_as_1d(x), eps)
    y = np.maximum(_as_1d(y), eps)
    lx, ly = np.log(x), np.log(y)
    a, b = np.polyfit(lx, ly, 1)
    corr = float(np.corrcoef(lx, ly)[0, 1])
    return {"slope": float(a), "intercept": float(b), "corr": corr}


# -----------------------
# Main notebook-friendly API
# -----------------------

def uq_parity_from_paths(
    *,
    val_path: str,
    test_path: str,
    ref_energy_key: str,
    pred_energy_key: str,
    pred_var_key: Optional[str] = None,
    pred_logvar_key: Optional[str] = None,
    pred_std_key: Optional[str] = None,
    per_atom: bool = True,
    n_bins: int = 15,
    min_count: int = 20,
    title: str = "UQ parity (RMV vs RMSE) on TEST",
    show_pre_calibration: bool = True,
    return_details: bool = True,
) -> Dict[str, object]:
    """
    Notebook workflow:
      1) Load val + test arrays
      2) Fit alpha^2 on validation
      3) Apply to test variances
      4) Bin test points by predicted variance (post-calibration)
      5) Compute ENCE + log-log stats on test
      6) Plot parity on test, optionally show pre-calibration points too

    Returns a dict of metrics + arrays.
    """
    # Load data
    yv, yvh, vv, _ = load_uq_from_extxyz(
        val_path,
        ref_energy_key=ref_energy_key,
        pred_energy_key=pred_energy_key,
        pred_var_key=pred_var_key,
        pred_logvar_key=pred_logvar_key,
        pred_std_key=pred_std_key,
        per_atom=per_atom,
    )
    yt, yth, vt, _ = load_uq_from_extxyz(
        test_path,
        ref_energy_key=ref_energy_key,
        pred_energy_key=pred_energy_key,
        pred_var_key=pred_var_key,
        pred_logvar_key=pred_logvar_key,
        pred_std_key=pred_std_key,
        per_atom=per_atom,
    )

    # Fit alpha^2 ONLY on validation
    alpha2 = fit_alpha2_on_validation(yv, yvh, vv)
    vt_cal = alpha2 * vt

    # Binning done on TEST (as you requested), using predicted variance
    # If you want to bin by *uncalibrated* variance, swap vt_cal -> vt below.
    bins, counts = make_bins(vt_cal, n_bins=n_bins, min_count=min_count)

    rmv_pre, rmse_pre = rmv_rmse_per_bin(yt, yth, vt, bins)          # pre-calibration
    rmv_post, rmse_post = rmv_rmse_per_bin(yt, yth, vt_cal, bins)    # post-calibration

    # Metrics computed on TEST (post-calibration is what you care about)
    ence_pre = ence(rmv_pre, rmse_pre)
    ence_post = ence(rmv_post, rmse_post)
    stats_pre = loglog_fit_stats(rmv_pre, rmse_pre)
    stats_post = loglog_fit_stats(rmv_post, rmse_post)

    # Print summary
    unit = "eV/atom" if per_atom else "eV"
    print("=== UQ parity (calibrate on VAL, evaluate on TEST) ===")
    print(f"VAL path:  {val_path}")
    print(f"TEST path: {test_path}")
    print(f"Quantity unit: {unit}")
    print(f"Bins used: {len(counts)} (requested n_bins={n_bins}, min_count={min_count})")
    print(f"alpha^2 (fit on VAL): {alpha2:.6g}")
    print("--- TEST metrics ---")
    print(f"ENCE pre-cal:  {ence_pre:.6g}")
    print(f"ENCE post-cal: {ence_post:.6g}")
    print(f"log-log pre:  slope={stats_pre['slope']:.4f}, corr={stats_pre['corr']:.4f}")
    print(f"log-log post: slope={stats_post['slope']:.4f}, corr={stats_post['corr']:.4f}")

    # Plot
    plt.figure(figsize=(6.4, 5.4), dpi=130)

    # point size ~ bin count
    sizes = 40 * (counts / np.max(counts))

    if show_pre_calibration:
        plt.scatter(rmv_pre, rmse_pre, s=sizes, alpha=0.45, label="pre-cal (test)")

    plt.scatter(rmv_post, rmse_post, s=sizes, alpha=0.85, label="post-cal (test)")

    # parity line
    eps = 1e-12
    lo = max(min(np.min(rmv_post), np.min(rmse_post)), eps)
    hi = max(np.max(rmv_post), np.max(rmse_post))
    xs = np.array([lo, hi])
    plt.plot(xs, xs, linewidth=1.5, label="parity")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("RMV = sqrt(mean predicted variance)  [log]")
    plt.ylabel("RMSE = sqrt(mean squared residual)   [log]")
    plt.title(f"{title}\nalpha^2={alpha2:.3g} | ENCE={ence_post:.3g} | bins={len(counts)}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    out = {
        "alpha2": alpha2,
        "bins_used": int(len(counts)),
        "counts": counts,
        "rmv_pre": rmv_pre,
        "rmse_pre": rmse_pre,
        "rmv_post": rmv_post,
        "rmse_post": rmse_post,
        "ence_pre": ence_pre,
        "ence_post": ence_post,
        "loglog_pre": stats_pre,
        "loglog_post": stats_post,
    }
    if not return_details:
        # strip arrays if you only want metrics
        for k in ["counts", "rmv_pre", "rmse_pre", "rmv_post", "rmse_post"]:
            out.pop(k, None)
    return out


