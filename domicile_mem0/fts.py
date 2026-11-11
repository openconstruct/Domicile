# Keyword query builder + stopwords, copied from Domicile mem0_core.py (same FTS5 semantics).
import re

_FTS_STOPWORDS = {'the', 'not', 'about', 'they', 'how', 'are', 'as', 'by', 'in', 'you', 'to', 'at', 'when', 'is', 'does', 'or', 'a', 'of', 'were', 'why', 'with', 'he', 'did', 'where', 'i', 'we', 'she', 'it', 'which', 'an', 'do', 'that', 'and', 'from', 'its', 'was', 'what', 'on', 'there', 'be', 'here', 'for', 'this'}

def _fts_match_query(query: str) -> str | None:
    toks = re.findall(r"[A-Za-z0-9_]+", query.lower())
    seen, uniq = set(), []
    for t in toks:
        if len(t) < 2 or t in _FTS_STOPWORDS or t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    if not uniq:
        return None
    return " OR ".join(f'"{t}"' for t in uniq)
