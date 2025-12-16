#!/usr/bin/env python3
"""
filter_feed.py - simplified version with automatic GitHub titles fetch
"""

import os
import re
import hashlib
import feedparser
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from dateutil import parser as dateparser
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import requests

# ===== CONFIG =====
FEEDS_FILE = "feeds.txt"
REFERENCE_FILE = "titles.txt"
REFERENCE_URL = "https://raw.githubusercontent.com/evilgodfahim/ref/main/titles.txt"
REF_EMB_NPY = "ref_embeddings.npy"
OUTPUT_FILE = "filtered.xml"

MODEL_PATH = "models/all-mpnet-base-v2"
USE_SMALL_MODEL = False

ENGLISH_SIM_THRESHOLD = 0.60
HYBRID_MIN_SIM_LOW = 0.33
HYBRID_MIN_SIM_HIGH = 0.38
HYBRID_PATTERN_MED = 7
HYBRID_PATTERN_HIGH = 8

CUTOFF_HOURS = 36
MAX_OUTPUT_ITEMS = 50

DBSCAN_EPS = 0.25
DBSCAN_MIN_SAMPLES = 1

# ---- New single control parameter (0.0 lenient -> 1.0 strict) ----
FILTER_STRENGTH = 0.66  # default moderate
DEBUG_FILTER = False    # set True to print per-article debug info
# -------------------------------------------------------------------


# ===== UTIL =====
def clean_title(t: str) -> str:
    if not t:
        return ""
    t = t.strip()
    t = re.sub(r'["""\'`]', "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def parse_iso_to_utc_naive(s: str):
    try:
        dt = dateparser.parse(s)
        if dt is None:
            return None
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None

def timestamp_from_pubdate(s: str) -> int:
    dt = parse_iso_to_utc_naive(s)
    if dt is None:
        return 0
    return int(dt.timestamp())

def domain_from_url(url: str) -> str:
    try:
        m = re.search(r"https?://([^/]+)/?", url)
        if m:
            return m.group(1).lower()
    except Exception:
        pass
    return ""


# ===== FETCH TITLES FROM GITHUB =====
def fetch_reference_titles():
    try:
        print("[fetch_reference_titles] fetching titles.txt from GitHub...")
        r = requests.get(REFERENCE_URL, timeout=15)
        if r.status_code == 200:
            with open(REFERENCE_FILE, "w", encoding="utf-8") as f:
                f.write(r.text)
            print("[fetch_reference_titles] ✓ downloaded titles.txt")
        else:
            print(f"[fetch_reference_titles] ERROR: status {r.status_code}")
    except Exception as e:
        print(f"[fetch_reference_titles] ERROR: {e}")


# ===== LOAD REFERENCE TITLES =====
def load_reference_titles():
    if not os.path.exists(REFERENCE_FILE):
        print(f"[load_reference_titles] WARNING: {REFERENCE_FILE} not found")
        return []
    try:
        with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
            titles = [clean_title(line) for line in f if clean_title(line)]
        print(f"[load_reference_titles] ✓ loaded {len(titles)} titles from {REFERENCE_FILE}")
        return titles
    except Exception as e:
        print(f"[load_reference_titles] ERROR: {e}")
        return []


# ===== PATTERN SCORING =====
def calculate_analytical_score(title: str) -> int:
    if not title:
        return 0
    tl = title.lower()
    score = 0

    if any(p in tl for p in ['-', ' v ', ' vs ', ' versus ', ' and ', ' with ']):
        score += 2
    if "'s " in tl or "'" in tl:
        score += 2

    question_starters = ['why ','how ','what ','can ','will ','should ','is ','are ','do ','does ','could ','may ','might ','would ']
    if any(tl.startswith(q) for q in question_starters):
        score += 2

    return score


# ===== MAIN =====
def main():
    print("[main] starting filter_feed.py")

    # Fetch reference titles
    fetch_reference_titles()
    ref_titles = load_reference_titles()

    # Load model
    model_name = "sentence-transformers/all-MiniLM-L6-v2" if USE_SMALL_MODEL else MODEL_PATH
    if not os.path.exists(model_name) and model_name == MODEL_PATH:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = SentenceTransformer(model_name)

    # ---- Compute/load reference embeddings (robust: keyed by titles file hash) ----
    ref_embeddings = None
    if ref_titles:
        def _titles_hash(titles_list):
            h = hashlib.sha256()
            joined = "\n".join(titles_list).encode("utf-8")
            h.update(joined)
            return h.hexdigest()

        titles_hash = _titles_hash(ref_titles)
        hash_file = REF_EMB_NPY + ".sha256"

        if os.path.exists(REF_EMB_NPY) and os.path.exists(hash_file):
            try:
                with open(hash_file, "r", encoding="utf-8") as hf:
                    cached_hash = hf.read().strip()
                if cached_hash == titles_hash:
                    try:
                        emb_cache = np.load(REF_EMB_NPY)
                        if emb_cache.size > 0:
                            ref_embeddings = emb_cache
                            print(f"[ref_emb] loaded cached embeddings ({REF_EMB_NPY})")
                    except Exception as e:
                        print(f"[ref_emb] failed loading cache, will recompute: {e}")
                        ref_embeddings = None
                else:
                    print("[ref_emb] titles changed: regenerating reference embeddings")
            except Exception as e:
                print(f"[ref_emb] failed reading hash file: {e}")

        if ref_embeddings is None:
            try:
                print("[ref_emb] computing reference embeddings...")
                ref_embeddings = model.encode(ref_titles, convert_to_numpy=True)
                np.save(REF_EMB_NPY, ref_embeddings)
                with open(hash_file, "w", encoding="utf-8") as hf:
                    hf.write(titles_hash)
                print(f"[ref_emb] saved embeddings to {REF_EMB_NPY} and hash to {hash_file}")
            except Exception as e:
                print(f"[ref_emb] ERROR computing/saving embeddings: {e}")

    # Load feeds
    if not os.path.exists(FEEDS_FILE):
        raise SystemExit(f"{FEEDS_FILE} missing")
    with open(FEEDS_FILE, "r", encoding="utf-8") as f:
        feed_urls = [l.strip() for l in f if l.strip()]

    feed_articles = []
    for url in feed_urls:
        try:
            # ===== Minimal change: fetch feed via requests with User-Agent =====
            headers = {"User-Agent": "Mozilla/5.0 (compatible; MyBot/1.0)"}
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                print(f"[main] WARNING: failed to fetch {url}, status {resp.status_code}")
                continue
            feed = feedparser.parse(resp.content)
        except Exception as e:
            print(f"[main] ERROR fetching/parsing feed {url}: {e}")
            continue

        for e in feed.entries:
            if getattr(e, "title", None) and getattr(e, "link", None):
                feed_articles.append({
                    "title": clean_title(e.title),
                    "link": e.link,
                    "published": getattr(e, "published", "") or getattr(e, "updated", ""),
                    "feed_source": url
                })

    if not feed_articles:
        print("[main] no feed articles found; exiting")
        return

    # Encode articles
    article_emb = model.encode([a["title"] for a in feed_articles], convert_to_numpy=True)

    # ===== HYBRID FILTER (ENHANCED) =====
    candidates = []

    TOP_K = 5
    for idx, a in enumerate(feed_articles):
        t = a["title"]
        pat = calculate_analytical_score(t)
        art_emb = article_emb[idx]

        max_sim = 0.0
        mean_topk = 0.0
        ref_coherence = 0.0
        if ref_embeddings is not None and ref_embeddings.size > 0:
            sims = cosine_similarity([art_emb], ref_embeddings)[0]
            max_sim = float(np.max(sims))
            k = min(TOP_K, sims.shape[0])
            topk_idx = np.argsort(sims)[-k:][::-1]
            topk_vals = sims[topk_idx] if topk_idx.size > 0 else np.array([max_sim])
            mean_topk = float(np.mean(topk_vals)) if topk_vals.size > 0 else max_sim
            if len(topk_idx) > 1:
                topk_embs = ref_embeddings[topk_idx]
                pair_mat = cosine_similarity(topk_embs)
                n = pair_mat.shape[0]
                if n > 1:
                    upper_ix = np.triu_indices(n, k=1)
                    ref_coherence = float(np.mean(pair_mat[upper_ix])) if upper_ix[0].size > 0 else 0.0
                else:
                    ref_coherence = 0.0
            else:
                ref_coherence = 0.0

        norm_pat = max(0.0, min(1.0, pat / 6.0))

        shallow_penalty = 0.0
        if t:
            tl = t.strip()
            if tl.isupper():
                shallow_penalty += 0.45
            if "!" in tl:
                shallow_penalty += 0.25
            low_quality_phrases = ["you won't believe", "won't believe", "shocking", "what happened next",
                                   "this is why", "must see", "can't miss", "will blow your mind"]
            tl_lower = tl.lower()
            if any(p in tl_lower for p in low_quality_phrases):
                shallow_penalty += 0.35
            if re.search(r'^\d+\s+(ways|things|reasons)\b', tl_lower):
                shallow_penalty += 0.20
        shallow_penalty = min(1.0, shallow_penalty)

        w_max, w_mean, w_coh, w_pat = 0.60, 0.20, 0.10, 0.10
        composite = (w_max * max_sim) + (w_mean * mean_topk) + (w_coh * ref_coherence) + (w_pat * norm_pat)
        composite = composite * (1.0 - 0.65 * shallow_penalty)

        adaptive_threshold = 0.35 + FILTER_STRENGTH * 0.45
        min_sim_required = 0.20 + FILTER_STRENGTH * 0.45
        pat_thresh = 1 + int(FILTER_STRENGTH * 4)
        accept = False
        if composite >= adaptive_threshold:
            if max_sim >= min_sim_required or pat >= pat_thresh:
                accept = True

        if DEBUG_FILTER:
            print(f"[DEBUG] title={t!r} max_sim={max_sim:.3f} mean_topk={mean_topk:.3f} "
                  f"ref_coh={ref_coherence:.3f} pat={pat} shallow_pen={shallow_penalty:.2f} "
                  f"composite={composite:.3f} adaptive_th={adaptive_threshold:.3f} accept={accept}")

        if accept:
            meta = a.copy()
            meta.update({
                "pattern_score": pat,
                "sim_to_refs": max_sim,
                "embedding": art_emb,
                "timestamp": timestamp_from_pubdate(a.get("published",""))
            })
            candidates.append(meta)

    if not candidates:
        print("[main] no candidates passed filter; exiting")
        return

    # ===== TIME CUTOFF =====
    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(hours=CUTOFF_HOURS)).timestamp())
    candidates = [c for c in candidates if c["timestamp"] >= cutoff_ts]
    print(f"[main] {len(candidates)} candidates after {CUTOFF_HOURS}h cutoff")
    if not candidates:
        print("[main] no recent candidates; exiting")
        return

    # ===== CLUSTERING (DBSCAN) =====
    X = np.vstack([c["embedding"] for c in candidates])
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine").fit(X)
    labels = clustering.labels_

    for lbl, c in zip(labels, candidates):
        c["cluster"] = int(lbl)

    clusters = {}
    for c in candidates:
        clusters.setdefault(c["cluster"], []).append(c)

    def cluster_score(cluster):
        size = len(cluster)
        avg_pat = sum(x["pattern_score"] for x in cluster) / max(1, size)
        max_ts = max(x["timestamp"] for x in cluster)
        age_hours = max(1, (datetime.now(timezone.utc).timestamp() - max_ts) / 3600.0)
        recency_boost = 1.0 / (1.0 + age_hours / 24.0)
        return size * (1 + avg_pat / 5.0) * (1 + recency_boost)

    cluster_list = []
    for lbl, items in clusters.items():
        cluster_list.append({"label": lbl, "items": items, "score": cluster_score(items)})
    cluster_list.sort(key=lambda x: x["score"], reverse=True)

    # ===== DIVERSITY & FINAL SELECTION =====
    def compute_item_score(it):
        recency_hours = max(1, (datetime.now(timezone.utc).timestamp() - it["timestamp"]) / 3600.0)
        recency_score = 1.0 / (1.0 + recency_hours / 24.0)
        return it["pattern_score"] * 2.0 + it["sim_to_refs"] * 5.0 + recency_score

    cluster_reps = []
    for cl in cluster_list:
        items = cl["items"]
        best_item = None
        best_score = -1
        for it in items:
            sc = compute_item_score(it)
            if sc > best_score:
                best_score = sc
                best_item = it
        if best_item is not None:
            cluster_reps.append({"label": cl["label"], "rep": best_item, "cluster_score": cl["score"], "rep_item_score": best_score})

    cluster_reps.sort(key=lambda x: (x["cluster_score"], x["rep_item_score"]), reverse=True)

    final = []
    seen_titles = set()
    for crep in cluster_reps:
        art = crep["rep"]
        if art["title"] in seen_titles:
            continue
        final.append(art)
        seen_titles.add(art["title"])
        if len(final) >= MAX_OUTPUT_ITEMS:
            break

    if not final:
        for cl in cluster_list[:MAX_OUTPUT_ITEMS]:
            final.append(cl["items"][0])

    print(f"[main] selected {len(final)} final articles")

    # ===== APPEND MODE: KEEP LAST 500 ITEMS =====
    print("[main] appending to existing filtered.xml (max 500 items)")

    existing_items = []
    if os.path.exists(OUTPUT_FILE):
        try:
            tree_prev = ET.parse(OUTPUT_FILE)
            root_prev = tree_prev.getroot()
            for item in root_prev.findall("./channel/item"):
                existing_items.append({
                    "title": (item.findtext("title") or "").strip(),
                    "link": (item.findtext("link") or "").strip(),
                    "pubDate": (item.findtext("pubDate") or "").strip(),
                    "source": (item.findtext("source") or "").strip()
                })
            print(f"[main] loaded {len(existing_items)} existing items from {OUTPUT_FILE}")
        except Exception as e:
            print(f"[main] warning: failed to parse existing xml: {e}")

    seen = set()
    merged = []

    def _norm_key(title: str, link: str) -> str:
        if link:
            return link.strip()
        return re.sub(r"\s+", " ", (title or "").strip().lower())

    for art in final:
        key = _norm_key(art.get("title",""), art.get("link",""))
        if not key:
            continue
        if key not in seen:
            seen.add(key)
            merged.append({
                "title": art.get("title",""),
                "link": art.get("link",""),
                "pubDate": datetime.utcfromtimestamp(art.get("timestamp", int(datetime.now(timezone.utc).timestamp()))).strftime("%a, %d %b %Y %H:%M:%S GMT"),
                "source": art.get("feed_source", "")
            })

    for art in existing_items:
        key = _norm_key(art.get("title",""), art.get("link",""))
        if not key:
            continue
        if key not in seen:
            seen.add(key)
            merged.append(art)

    MAX_ARCHIVE_ITEMS = 500
    merged = merged[:MAX_ARCHIVE_ITEMS]
    print(f"[main] merged total items after dedupe: {len(merged)} (capped at {MAX_ARCHIVE_ITEMS})")

    root = ET.Element("rss", version="2.0")
    channel = ET.SubElement(root, "channel")
    ET.SubElement(channel, "title").text = "Filtered Feed"
    ET.SubElement(channel, "link").text = "https://example.com"
    ET.SubElement(channel, "description").text = "Filtered feed articles"

    for art in merged:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = art.get("title", "")
        ET.SubElement(item, "link").text = art.get("link", "")
        ET.SubElement(item, "pubDate").text = art.get("pubDate", "")
        ET.SubElement(item, "source").text = art.get("source", "")

    tree = ET.ElementTree(root)
    tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)
    print(f"[main] appended {len(final)} new items; final {len(merged)} items written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()