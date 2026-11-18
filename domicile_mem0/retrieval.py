# Retrieval tuning shared by every store: flash memory (mem0_v2) and full
# memory (mem2_server). ONE copy: these used to be pasted into each backend
# and had to be changed in lockstep.
#
# Embedder-specific values (HINT_THRESHOLD, GUARD_CUTOFF, EMBED_MODEL) are NOT
# here: they are calibrated per model and belong to the backend that uses it.
import hashlib
import math
from datetime import date

RRF_K = 60  # reciprocal-rank-fusion damping; 60 is the standard default

HINT_LIMIT = 3
HINT_RECENCY_BONUS = 0.06
HINT_RECENCY_DAYS = 64

HYBRID_RECENCY_BONUS = 0.6 / RRF_K
HYBRID_RECENCY_DAYS = HINT_RECENCY_DAYS

# salience: "used stays, unused fades"
HINT_STALE_FLOOR = -0.10     # cosine-domain adj at full staleness
HINT_STALE_DAYS = 180        # linear 0 (@ HINT_RECENCY_DAYS) -> floor (here)
HINT_FREQ_CAP = 0.12         # max buy-back from frequent explicit use
HINT_FREQ_K = 0.03           # geometric: min(cap, k * log2(days_used + 1))
EVENT_WEIGHT = 0.25

_HYBRID_SCALE = HYBRID_RECENCY_BONUS / HINT_RECENCY_BONUS
HYBRID_STALE_FLOOR = HINT_STALE_FLOOR * _HYBRID_SCALE
HYBRID_STALE_DAYS = HINT_STALE_DAYS
HYBRID_FREQ_CAP = HINT_FREQ_CAP * _HYBRID_SCALE
HYBRID_FREQ_K = HINT_FREQ_K * _HYBRID_SCALE


def recency_bonus(created_at, peak=HINT_RECENCY_BONUS, horizon=HINT_RECENCY_DAYS):
    """`peak` today, stepping down by log2(age in days) to 0 at `horizon`."""
    try:
        age = (date.today() - date.fromisoformat((created_at or "")[:10])).days
    except ValueError:
        return 0.0
    if age <= 0:
        return peak
    if age >= horizon:
        return 0.0
    step = peak / max(1.0, math.log2(horizon))
    return max(0.0, peak - step * math.floor(math.log2(age)))


def salience_adjust(last_used, created_at, use_count, *, peak, horizon, floor,
                    stale_days, freq_cap, freq_k, event_weight=EVENT_WEIGHT):
    """Score adjustment for "used stays, unused fades". Two summed terms:
    a decay on last use / creation date, and a capped bonus for use count."""
    used_stamp = last_used or created_at
    try:
        age = (date.today() - date.fromisoformat((used_stamp or "")[:10])).days
    except ValueError:
        age = 0
    if age <= horizon:
        used_decay = recency_bonus(used_stamp, peak=peak, horizon=horizon)
        created_decay = recency_bonus(created_at, peak=peak, horizon=horizon)
        decay = (1 - event_weight) * used_decay + event_weight * created_decay
    else:
        frac = min(1.0, (age - horizon) / max(1, stale_days - horizon))
        decay = floor * frac
    try:
        uc = max(0, int(use_count or 0))
    except (TypeError, ValueError):
        uc = 0
    freq = min(freq_cap, freq_k * math.log2(uc + 1)) if uc else 0.0
    return decay + freq


def chunk_digest(text):
    """Idempotency key for a verbatim chunk."""
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()[:16]
