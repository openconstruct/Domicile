#!/usr/bin/env python3
"""Dedup guard: calibration and checks for the memory extraction pipeline."""

import numpy as np

DEFAULT_CUTOFF = 0.70


def _nearest_cos(store, text, user_id, exclude_id=None, top_k=5):
    try:
        vec = store.embedding_model.embed(text, "search")
    except Exception:
        return 0.0
    res = store.vector_store.search(
        query=text,
        vectors=vec,
        top_k=top_k,
        filters={"user_id": user_id},
    )
    best = 0.0
    for r in res:
        rid = getattr(r, "id", None)
        if exclude_id is not None and rid == exclude_id:
            continue
        score = getattr(r, "score", 0.0) or 0.0
        if score > best:
            best = score
    return best


def guard_new_facts(store, added_rows, user_id, cutoff=DEFAULT_CUTOFF):
    kept = []
    for row in added_rows:
        mid = row.get("id")
        text = row.get("memory") or row.get("data") or ""
        if not text:
            continue
        near = _nearest_cos(store, text, user_id, exclude_id=mid, top_k=5)
        if near >= cutoff:
            try:
                store.delete(memory_id=mid)
            except Exception as e:
                kept.append(mid)
                continue
            kept.append(None)  # marker: deleted
        else:
            kept.append(mid)
    return [k for k in kept if k]


def calibrate_corpus():
    """Synthetic ground truth for cutoff tuning (invented facts about invented people;
    nothing here describes a real install). Three buckets:"""
    dup_clusters = [
        [
            "Sam decided to force the nightly ingest to run, then assess the damage afterward.",
            "Sam chose to force ingest and then assess what broke.",
            "Decision: run the ingest by force now; evaluate the fallout later.",
        ],
        [
            "The nightly timer died on 2026-03-02 at 01:22 and never recovered on its own.",
            "Nightly ingest stopped at 01:22 on Mar 2 and stayed down without restart.",
            "The scheduled nightly job failed at 01:22 2026-03-02 and did not auto-recover.",
        ],
        [
            "Model A won the extraction bake-off because it carries the why behind decisions.",
            "Bake-off result: model A preferred for permanent store; it preserves reasoning, not just conclusions.",
            "Verdict: use model A for the memory rebuild -- it abstracts to the right altitude and keeps the rationale.",
        ],
    ]
    related = [
        "The vendor endpoint serves /responses only, not /chat/completions.",
        "The ambient hint now truncates each memory to a 20-word snippet before the turn.",
        "The home server is a small fanless mini PC; the gaming laptop does the inference.",
        "The dedup guard reuses mem0's own embedding_model instance to match store distances.",
    ]
    distinct = [
        "Pat's sister lives in Denver and teaches high-school chemistry.",
        "The workstation plan pairs two used GPUs with a server board and 128GB of RAM.",
        "Pat started a new medication in the spring and sleeps better since.",
        "The side project is a small language model fine-tuned on forum posts.",
    ]
    out = []
    for c in dup_clusters:
        out.append(("DUP", c[0]))
        for t in c[1:]:
            out.append(("DUP", t))
    for t in related:
        out.append(("REL", t))
    for t in distinct:
        out.append(("DIS", t))
    dup_ids = []
    for ci, c in enumerate(dup_clusters):
        for _ in c:
            dup_ids.append(ci)
    return out, dup_ids


def run_calibration(cutoffs=None):
    import paths  # noqa: F401  (sets FASTEMBED_CACHE_PATH)
    from fastembed import TextEmbedding

    if cutoffs is None:
        cutoffs = [round(0.80 + 0.01 * i, 2) for i in range(16)]  # 0.80..0.95

    corpus, dup_ids = calibrate_corpus()
    labels = [l for l, _ in corpus]
    texts = [t for _, t in corpus]

    emb = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vecs = np.array([np.array(list(emb.embed(t)))[0] for t in texts])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(norms, 1e-9, None)
    cos = vecs @ vecs.T

    n = len(texts)
    results = []
    for cut in cutoffs:
        tp = fp = fn = tn = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = cos[i, j]
                same_dup_cluster = (
                    labels[i] == "DUP" and labels[j] == "DUP"
                    and dup_ids[i] == dup_ids[j]
                )
                merged = c >= cut
                if same_dup_cluster:
                    if merged:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if merged:
                        fp += 1
                    else:
                        tn += 1
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        results.append((cut, prec, rec, f1, tp, fp, fn, tn))
    return results


if __name__ == "__main__":
    res = run_calibration()
    print(f"{'cut':>5} {'prec':>5} {'rec':>5} {'f1':>5}  (tp/fp/fn/tn)")
    for cut, p, r, f, tp, fp, fn, tn in res:
        print(f"{cut:>5} {p:>5.2f} {r:>5.2f} {f:>5.2f}  ({tp}/{fp}/{fn}/{tn})")
    best = max(res, key=lambda x: x[3])
    print(f"\nF1-optimal cutoff: {best[0]} (prec={best[1]:.2f} rec={best[2]:.2f} f1={best[3]:.2f})")
