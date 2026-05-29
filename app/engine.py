"""
Social Media Monitoring Agent — Inference Engine
================================================
Pure-Python inference layer shared by the web server (app/server.py).
Holds the retrieval / sentiment / clustering / summarisation logic that
used to live inside the Streamlit app, decoupled from any UI framework.
"""

from __future__ import annotations

import pickle
import uuid
import warnings
from pathlib import Path
from threading import Lock

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
INDEX_DIR = BASE_DIR / "data" / "index"
MODEL_DIR = BASE_DIR / "data" / "models"

EMOTIONS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "love", "optimism", "pessimism", "sadness", "surprise", "trust",
]
POSITIVE_EMOTIONS = {"joy", "love", "optimism", "anticipation", "trust"}
NEGATIVE_EMOTIONS = {"anger", "disgust", "fear", "pessimism", "sadness"}

ICON_POLARITY = {"positive": "😊", "negative": "😠", "neutral": "😐"}

# ── Lazy global resources ────────────────────────────────────────────────────
_RES = None
_RES_LOCK = Lock()
_SUMMARIZER = None
_SUMMARIZER_LOCK = Lock()

# Cache of recent search results so the summary endpoint can reuse embeddings
# without re-running retrieval. Keyed by a uuid handed back to the client.
_RESULT_CACHE: dict[str, dict] = {}
_RESULT_ORDER: list[str] = []
_CACHE_MAX = 8


def load_resources():
    """Load corpus, BM25, embeddings, FAISS index and the embedder once."""
    global _RES
    if _RES is not None:
        return _RES
    with _RES_LOCK:
        if _RES is not None:
            return _RES

        corpus = pd.read_parquet(PROC_DIR / "tweets_processed.parquet").reset_index(drop=True)
        with open(INDEX_DIR / "bm25.pkl", "rb") as f:
            bm25: BM25Okapi = pickle.load(f)
        all_embeddings = np.load(INDEX_DIR / "embeddings.npy")
        faiss_index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Precompute ID → row index for fast embedding lookup.
        id_to_idx = {rid: i for i, rid in enumerate(corpus["ID"].tolist())}

        _RES = {
            "corpus": corpus,
            "bm25": bm25,
            "embeddings": all_embeddings,
            "faiss": faiss_index,
            "embedder": embedder,
            "id_to_idx": id_to_idx,
        }
    return _RES


def load_summarizer():
    """Lazily load the abstractive summarisation model (~600 MB)."""
    global _SUMMARIZER
    if _SUMMARIZER is not None:
        return _SUMMARIZER
    with _SUMMARIZER_LOCK:
        if _SUMMARIZER is not None:
            return _SUMMARIZER
        import torch
        from transformers import pipeline as hf_pipeline

        device_id = 0 if torch.cuda.is_available() else -1
        _SUMMARIZER = hf_pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=device_id,
            truncation=True,
        )
    return _SUMMARIZER


# ══════════════════════════════════════════════════════════════════════════════
# Search
# ══════════════════════════════════════════════════════════════════════════════

def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)


def bm25_search(query, bm25, corpus, top_k):
    tokens = query.lower().split()
    scores = np.array(bm25.get_scores(tokens))
    idxs   = np.argsort(scores)[::-1][:top_k]
    df = corpus.iloc[idxs].copy()
    df["hybrid_score"] = scores[idxs]
    return df.reset_index(drop=True)


def semantic_search(query, embedder, faiss_index, corpus, top_k):
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    dists, idxs = faiss_index.search(q_emb, top_k)
    df = corpus.iloc[idxs[0]].copy()
    df["hybrid_score"] = dists[0]
    return df.reset_index(drop=True)


def hybrid_search(query, bm25, embedder, faiss_index, corpus, top_k=200, alpha=0.5):
    tokens  = query.lower().split()
    bm25_sc = np.array(bm25.get_scores(tokens))

    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    sem_sc = np.zeros(len(corpus))
    dists, idxs = faiss_index.search(q_emb, len(corpus))
    sem_sc[idxs[0]] = dists[0]

    fused = alpha * _minmax(bm25_sc) + (1 - alpha) * _minmax(sem_sc)
    top_idxs = np.argsort(fused)[::-1][:top_k]

    df = corpus.iloc[top_idxs].copy()
    df["hybrid_score"] = fused[top_idxs]
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Analytics
# ══════════════════════════════════════════════════════════════════════════════

def assign_polarity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _pol(row):
        pos = sum(row.get(e, 0) for e in POSITIVE_EMOTIONS)
        neg = sum(row.get(e, 0) for e in NEGATIVE_EMOTIONS)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"

    df["polarity"] = df.apply(_pol, axis=1)
    return df


def cluster_umap(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """K-Means (k≤4) + UMAP 2-D projection on the retrieved embeddings."""
    from sklearn.cluster import KMeans
    import umap

    df = df.copy()
    n = len(df)
    if n < 3:
        df["cluster"] = "0"
        df["umap_x"] = 0.0
        df["umap_y"] = 0.0
        return df

    k = min(4, n)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(embeddings)

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, n - 1))
    coords  = reducer.fit_transform(embeddings)

    df["cluster"] = labels.astype(str)
    df["umap_x"]  = coords[:, 0]
    df["umap_y"]  = coords[:, 1]
    return df


def top_terms_per_cluster(df: pd.DataFrame, n: int = 5) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer

    result = {}
    for cl in sorted(df["cluster"].unique()):
        texts = df[df["cluster"] == cl]["text_clean"].tolist()
        if not texts:
            continue
        try:
            tfidf = TfidfVectorizer(max_features=500, stop_words="english")
            X     = tfidf.fit_transform(texts)
            mean  = X.mean(axis=0).A1
            terms = [tfidf.get_feature_names_out()[i] for i in mean.argsort()[::-1][:n]]
            result[cl] = ", ".join(terms)
        except Exception:
            result[cl] = "—"
    return result


def extractive_summary(df_group: pd.DataFrame, emb_group: np.ndarray, top_n: int = 3):
    if len(df_group) == 0:
        return pd.DataFrame()
    centroid = emb_group.mean(axis=0, keepdims=True)
    sims     = cosine_similarity(emb_group, centroid).flatten()
    top_idxs = np.argsort(sims)[::-1][:top_n]
    out      = df_group.iloc[top_idxs].copy()
    out["centroid_sim"] = sims[top_idxs]
    return out


def abstractive_summary(summarizer, df_group: pd.DataFrame, query: str, top_n: int = 10) -> str:
    if len(df_group) == 0:
        return "(no tweets in this group)"
    top_tweets = (
        df_group.sort_values("hybrid_score", ascending=False)
        .head(top_n)["text_light"]
        .tolist()
    )
    prefix = f"Summarize the following tweets about {query}: "
    body   = " | ".join(top_tweets)
    prompt = (prefix + body)[:3200]
    try:
        res = summarizer(prompt, min_length=40, max_length=120, do_sample=False)
        return res[0]["summary_text"].strip()
    except Exception as e:
        return f"(summarisation error: {e})"


# ══════════════════════════════════════════════════════════════════════════════
# High-level orchestration (returns JSON-serialisable dicts)
# ══════════════════════════════════════════════════════════════════════════════

def _clean_tweet(t: str) -> str:
    return str(t).replace("\\n", " ").replace("\n", " ").strip()


def _cache_put(result_id: str, payload: dict):
    _RESULT_CACHE[result_id] = payload
    _RESULT_ORDER.append(result_id)
    while len(_RESULT_ORDER) > _CACHE_MAX:
        old = _RESULT_ORDER.pop(0)
        _RESULT_CACHE.pop(old, None)


def run_search(query: str, mode: str = "hybrid", alpha: float = 0.5, top_k: int = 200) -> dict:
    """Run retrieval + polarity + clustering and return a JSON-ready payload."""
    res = load_resources()
    corpus, bm25, embedder = res["corpus"], res["bm25"], res["embedder"]
    faiss_index, all_embeddings = res["faiss"], res["embeddings"]
    id_to_idx = res["id_to_idx"]

    mode = (mode or "hybrid").lower()
    if mode == "bm25":
        df = bm25_search(query, bm25, corpus, top_k)
    elif mode == "semantic":
        df = semantic_search(query, embedder, faiss_index, corpus, top_k)
    else:
        df = hybrid_search(query, bm25, embedder, faiss_index, corpus, top_k, alpha)

    df["rank"] = range(1, len(df) + 1)
    df = assign_polarity(df)

    ret_idxs = df["ID"].map(id_to_idx).values
    ret_embs = all_embeddings[ret_idxs]

    df = cluster_umap(df, ret_embs)
    cluster_terms = top_terms_per_cluster(df)

    result_id = uuid.uuid4().hex
    _cache_put(result_id, {"df": df, "ret_embs": ret_embs, "query": query})

    return _build_payload(result_id, query, df, ret_embs, cluster_terms)


def _build_payload(result_id, query, df, ret_embs, cluster_terms):
    n = len(df)

    # Polarity counts
    pol_counts = {p: int((df["polarity"] == p).sum()) for p in ["positive", "negative", "neutral"]}

    # Overall emotion activation rate
    emo_means = {e: float(df[e].mean()) for e in EMOTIONS}

    # Emotion heatmap per polarity (rows: polarity, cols: emotions)
    heatmap = {}
    for pol in ["positive", "negative", "neutral"]:
        sub = df[df["polarity"] == pol]
        heatmap[pol] = [float(sub[e].mean()) if len(sub) else 0.0 for e in EMOTIONS]

    # Tweet rows
    tweets = []
    for _, row in df.iterrows():
        active = [e for e in EMOTIONS if row.get(e, 0) == 1]
        tweets.append({
            "rank": int(row["rank"]),
            "tweet": _clean_tweet(row["Tweet"]),
            "polarity": row["polarity"],
            "score": round(float(row["hybrid_score"]), 4),
            "emotions": active,
        })

    # Cluster scatter points
    points = []
    for _, row in df.iterrows():
        points.append({
            "x": float(row["umap_x"]),
            "y": float(row["umap_y"]),
            "cluster": str(row["cluster"]),
            "polarity": row["polarity"],
            "tweet": _clean_tweet(row["Tweet"])[:160],
            "score": round(float(row["hybrid_score"]), 3),
        })

    # Aspect analysis (aspects derived from cluster top-terms)
    aspects = []
    for cl in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cl]
        rep = sub.sort_values("hybrid_score", ascending=False)
        aspects.append({
            "cluster": str(cl),
            "terms": cluster_terms.get(cl, "—"),
            "count": int(len(sub)),
            "polarity": {p: int((sub["polarity"] == p).sum()) for p in ["positive", "negative", "neutral"]},
            "emotions": [float(sub[e].mean()) if len(sub) else 0.0 for e in EMOTIONS],
            "rep_tweet": _clean_tweet(rep.iloc[0]["Tweet"]) if len(rep) else "",
            "rep_polarity": rep.iloc[0]["polarity"] if len(rep) else "neutral",
        })

    # Extractive summaries per polarity (fast — always computed)
    extractive = {}
    pol_values = df["polarity"].values
    for pol in ["positive", "negative", "neutral"]:
        mask = pol_values == pol
        sub = df[mask].reset_index(drop=True)
        sub_embs = ret_embs[mask]
        items = []
        if len(sub):
            ext = extractive_summary(sub, sub_embs, top_n=3)
            for _, r in ext.iterrows():
                active = [e for e in EMOTIONS if r.get(e, 0) == 1]
                items.append({
                    "tweet": _clean_tweet(r["Tweet"]),
                    "sim": round(float(r["centroid_sim"]), 3),
                    "emotions": active,
                })
        extractive[pol] = {"count": int(mask.sum()), "items": items}

    return {
        "result_id": result_id,
        "query": query,
        "count": n,
        "emotions_order": EMOTIONS,
        "polarity_counts": pol_counts,
        "emotion_means": emo_means,
        "heatmap": heatmap,
        "tweets": tweets,
        "points": points,
        "cluster_terms": {str(k): v for k, v in cluster_terms.items()},
        "aspects": aspects,
        "extractive": extractive,
    }


def run_abstractive(result_id: str) -> dict:
    """Generate abstractive summaries per polarity for a cached search result."""
    cached = _RESULT_CACHE.get(result_id)
    if cached is None:
        raise KeyError("result_id not found — please run a new search")

    df, query = cached["df"], cached["query"]
    summarizer = load_summarizer()

    out = {}
    for pol in ["positive", "negative", "neutral"]:
        sub = df[df["polarity"] == pol].reset_index(drop=True)
        out[pol] = abstractive_summary(summarizer, sub, query)
    return {"result_id": result_id, "abstractive": out}
