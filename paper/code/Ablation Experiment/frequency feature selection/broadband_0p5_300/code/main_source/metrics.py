"""Full-record correlations and amplitude errors."""
import numpy as np
from config import SCORED_FINGERS


def score(truth, prediction):
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 5:
        raise ValueError("Truth and prediction must have identical (time, 5) shapes.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite values in evaluation arrays.")
    a = y - y.mean(0)
    b = p - p.mean(0)
    r = (a * b).sum(0) / np.maximum(np.sqrt((a * a).sum(0) * (b * b).sum(0)), 1e-12)
    error = p - y
    return dict(r_fingers=r.tolist(), r4=float(r[SCORED_FINGERS].mean()), r5=float(r.mean()),
                mae4=float(np.abs(error[:, SCORED_FINGERS]).mean()),
                mse4=float(np.square(error[:, SCORED_FINGERS]).mean()), samples=len(y))
