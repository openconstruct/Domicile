# Ranking. This module is the calibration surface: every score
# MemoryCore returns flows through rank_scores, so recalibration touches
# exactly one function.
import time

# Calibrated to the OLD Qdrant scorer's scale. Our scorer shifts scores;
# recalibrate before trusting this in the hint path.
HINT_THRESHOLD = 0.66


def rank_scores(sims, salience, created_at, use_count, now=None):
    import numpy as np
    now = now or time.time()
    age = np.maximum(0.0, (now - np.nan_to_num(created_at, nan=now)) / 86400.0)
    recency = np.where(np.isnan(created_at), 1.0, 0.5 + 0.5 * np.exp(-age / 365.0))
    use_boost = 1.0 + np.minimum(use_count, 50) * 0.002
    return sims * (0.85 + 0.15 * salience * recency) * use_boost
