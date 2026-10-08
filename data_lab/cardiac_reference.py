"""Timestamped cardiac-window contract. PPG produces PRV, never measured ECG HRV.

No EEG proxies, stale-value carry, interval interpolation or differences across gaps.
The raw-signal detector is an experimental baseline, not validated Athena metrology.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, find_peaks

CONTRACT = "cardiac-window-v1"
EXTRACTOR = "causal-polarity-peaks-adjacent-intervals-v2"
SOURCES = {"ecg_rr", "ppg_prv"}


def empty(source, start, end, reason):
    return dict(contract=CONTRACT, extractor=EXTRACTOR, source=source,
                window_start=float(start), window_end=float(end), valid=False,
                rejection_reason=reason, beat_count=0, valid_interval_count=0,
                valid_pair_count=0, coverage=0.0, rejected_fraction=1.0,
                rate_bpm=None, rmssd_ms=None, log_rmssd=None)


def from_intervals(times, intervals_ms, source, start, end, valid=None):
    """times are interval END timestamps in seconds on the EEG clock.

    Retain every detected interval, including invalid ones, so adjacency is not lost.
    """
    result = empty(source, start, end, "invalid_input")
    t, rr = np.asarray(times, float), np.asarray(intervals_ms, float)
    if source not in SOURCES or not np.isfinite([start, end]).all() or end - start < 29:
        return result
    if t.ndim != 1 or rr.shape != t.shape or len(t) < 2 or not np.isfinite(t).all():
        return result
    if (np.diff(t) <= 0).any():
        result["rejection_reason"] = "nonmonotonic_or_duplicate_timestamps"; return result
    keep = (t <= end) & ((t - np.nan_to_num(rr, nan=0) / 1000) >= start)
    supplied = np.ones(len(t), bool) if valid is None else np.asarray(valid, bool)
    if supplied.shape != t.shape:
        return result
    t, rr, supplied = t[keep], rr[keep], supplied[keep]
    if len(t) < 2:
        result["rejection_reason"] = "insufficient_beats"; return result
    good = supplied & np.isfinite(rr) & (rr >= 300) & (rr <= 2000)
    # Local artifact screening never removes indices before computing adjacent differences.
    reference = np.asarray([np.nanmedian(rr[max(0, i-2):i+3]) for i in range(len(rr))])
    good &= np.abs(rr - reference) <= np.maximum(100, .25 * reference)
    adjacent = np.abs(np.diff(t) * 1000 - rr[1:]) <= np.maximum(20, .1 * rr[1:])
    pairs = good[:-1] & good[1:] & adjacent
    coverage = min(1.0, float(rr[good].sum() / (1000 * (end-start))))
    rejected = float(1 - good.mean())
    result.update(beat_count=len(t)+1, valid_interval_count=int(good.sum()),
                  valid_pair_count=int(pairs.sum()), coverage=coverage,
                  rejected_fraction=rejected)
    if not good.any() or end - t[good][-1] > 2.5:
        result["rejection_reason"] = "stale_or_no_beats"; return result
    if pairs.sum() < 20:
        result["rejection_reason"] = "insufficient_adjacent_pairs"; return result
    if coverage < .90 or rejected > .05:
        result["rejection_reason"] = "insufficient_clean_coverage"; return result
    rmssd = float(np.sqrt(np.mean(np.diff(rr)[pairs] ** 2)))
    if rmssd <= 0:
        result["rejection_reason"] = "zero_variability"; return result
    result.update(valid=True, rejection_reason=None, rate_bpm=float(60000 / rr[good].mean()),
                  rmssd_ms=rmssd, log_rmssd=float(np.log(rmssd)))
    return result


def from_signal(times, values, source, start, end):
    """Extract beats using only observations <= end; refuse acquisition gaps.

    An optional four-second causal warm-up may precede start. No future filtering.
    """
    result = empty(source, start, end, "invalid_signal")
    t, x = np.asarray(times, float), np.asarray(values, float)
    if source not in SOURCES or t.ndim != 1 or x.shape != t.shape or len(t) < 40:
        return result
    mask = (t >= start-4) & (t <= end)
    t, x = t[mask], x[mask]
    if len(t) < 40 or not np.isfinite(t).all() or not np.isfinite(x).all():
        return result
    dt = np.diff(t)
    if (dt <= 0).any():
        result["rejection_reason"] = "nonmonotonic_or_duplicate_timestamps"; return result
    fs = 1 / np.median(dt)
    if t[0] > start + 2/fs or t[-1] < end-2/fs or np.max(dt) > 3/fs:
        result["rejection_reason"] = "signal_gap_or_incomplete_window"; return result
    minimum_fs = 100 if source == "ecg_rr" else 20
    if fs < minimum_fs * .98 or np.std(x) == 0:
        result["rejection_reason"] = "unsupported_sampling_or_flat_signal"; return result
    low, high = (5, 35) if source == "ecg_rr" else (.5, 4)
    filtered = sosfilt(butter(2, [low, high], btype="bandpass", fs=fs, output="sos"), x-x[0])
    # ECG absolute-value detection counted background oscillations as extra beats
    # and alternated between R and S extrema. Choose one window-wide polarity,
    # then require a prominent upper-tail peak. This is still an experimental
    # amplitude detector, not an implementation of a validated QRS algorithm.
    detection = filtered
    height = None
    if source == "ecg_rr":
        stable = filtered[t >= start]
        upper, lower = np.percentile(stable, [99.5, .5])
        detection = filtered if upper >= -lower else -filtered
        height = .4 * max(upper, -lower)
    prominence = max(float(np.std(detection)) * .5, 1e-12)
    peaks, _ = find_peaks(detection, distance=max(1, int(.30*fs)), prominence=prominence, height=height)
    # Subsample parabolic refinement mitigates (but does not eliminate) pulse timing quantization.
    refined = []
    for k in peaks:
        if k == 0 or k == len(x)-1:
            continue
        denominator = detection[k-1] - 2*detection[k] + detection[k+1]
        offset = 0 if abs(denominator) < 1e-12 else .5*(detection[k-1]-detection[k+1])/denominator
        refined.append(float(t[k] + np.clip(offset, -.5, .5)/fs))
    peaks_t = np.asarray(refined)
    result = from_intervals(peaks_t[1:], np.diff(peaks_t)*1000, source, start, end)
    result.update(sampling_hz=float(fs), detector_metrology_validated=False)
    return result


def feature_vector(block, start, end, expected_source=None):
    """Strict model-input boundary; missing never means zero or neutral."""
    if not isinstance(block, dict) or block.get("contract") != CONTRACT or not block.get("valid"):
        raise ValueError("cardiac_unavailable")
    if block.get("extractor") != EXTRACTOR:
        raise ValueError("cardiac_extractor_mismatch_recompute_required")
    if block.get("source") not in SOURCES or (expected_source and block["source"] != expected_source):
        raise ValueError("cardiac_source_mismatch")
    numeric = [start, end, block.get("window_start"), block.get("window_end"),
               block.get("coverage"), block.get("rejected_fraction"), block.get("valid_pair_count")]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in numeric):
        raise ValueError("invalid_cardiac_metadata")
    if end-start < 29 or not 0 <= block["coverage"] <= 1 or not 0 <= block["rejected_fraction"] <= 1:
        raise ValueError("invalid_cardiac_metadata")
    if abs(block.get("window_start", float("inf"))-start) > .1 or abs(block.get("window_end", float("inf"))-end) > .1:
        raise ValueError("cardiac_window_mismatch")
    if block.get("valid_pair_count", 0) < 20 or block.get("coverage", 0) < .90 or block.get("rejected_fraction", 1) > .05:
        raise ValueError("cardiac_quality")
    x = np.asarray([block.get("rate_bpm"), block.get("log_rmssd")], float)
    if not np.isfinite(x).all() or not 30 <= x[0] <= 200:
        raise ValueError("invalid_cardiac_features")
    return x


def agreement(ecg, ppg):
    """Paired descriptive Bland-Altman statistics, not a hardware-validation certificate."""
    a, b = np.asarray(ecg, float), np.asarray(ppg, float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return {"n": int(ok.sum()), "reason": "insufficient_pairs"}
    d = b[ok]-a[ok]; bias = float(d.mean()); sd = float(d.std(ddof=1))
    return dict(n=int(ok.sum()), bias=bias, limits_of_agreement=[bias-1.96*sd, bias+1.96*sd],
                mae=float(np.abs(d).mean()), hardware_validation=False)
