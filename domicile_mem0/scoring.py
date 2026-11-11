# Ranking. This module is the calibration surface (Rook R4): every score
# the system returns flows through rank_score, so recalibration touches
# exactly one function.
import math
import time

# Calibrated to the OLD Qdrant scorer's scale. Our scorer shifts scores;
# recalibrate with Rook's calib_hint.py before trusting this in the hint path.
HINT_THRESHOLD = 0.66


def rank_score(similarity, salience=0.5, created_at=None, use_count=0, now=None):
    recency = 1.0
    if created_at:
        now = now or time.time()
        age_days = max(0.0, (now - created_at) / 86400.0)
        recency = 0.5 + 0.5 * math.exp(-age_days / 365.0)
    use_boost = 1.0 + min(use_count, 50) * 0.002
    return float(similarity) * (0.85 + 0.15 * salience * recency) * use_boost


def rank_scores(sims, salience, created_at, use_count, now=None):
    """Vectorized rank_score over numpy arrays -- identical formula."""
    import numpy as np
    now = now or time.time()
    age = np.maximum(0.0, (now - np.nan_to_num(created_at, nan=now)) / 86400.0)
    recency = np.where(np.isnan(created_at), 1.0, 0.5 + 0.5 * np.exp(-age / 365.0))
    use_boost = 1.0 + np.minimum(use_count, 50) * 0.002
    return sims * (0.85 + 0.15 * salience * recency) * use_boost
