"""MEGON I AI -- persistent memory: store, chunker, embeddings, hybrid index, retriever"""
from __future__ import annotations
from megon.base import *

# ==============================================================================
# 6. STORE -- the persistent memory (SQLite, append-only, deduped)
# ==============================================================================
def simhash64(tokens: Sequence[str]) -> int:
    """64-bit SimHash -> near-duplicate detection without storing anything extra.

    Returned as a *signed* 64-bit int because SQLite INTEGER is signed and will
    reject values >= 2**63.
    """
    v = [0] * 64
    for t in tokens:
        h = blake_int(t)
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= (1 << i)
    return out - (1 << 64) if out >= (1 << 63) else out


def hamming64(a: int, b: int) -> int:
    return bin((a ^ b) & ((1 << 64) - 1)).count("1")


class Store:
    def __init__(self, c: sqlite3.Connection):
        self.c = c

    # --- ingest ---------------------------------------------------------------
    def has_content(self, digest: str) -> bool:
        return self.c.execute("SELECT 1 FROM docs WHERE sha256=?", (digest,)).fetchone() is not None

    def near_duplicate(self, sh: int, thresh: int = 5) -> Optional[int]:
        """Cheap near-dup check against the most recent 4000 docs."""
        rows = self.c.execute("SELECT id, simhash FROM docs ORDER BY id DESC LIMIT 4000").fetchall()
        for r in rows:
            if r["simhash"] is not None and hamming64(sh, r["simhash"]) <= thresh:
                return r["id"]
        return None

    def add_doc(self, url: str, title: str, text: str, source_id: str, kind: str,
                split: str = "train", meta: Optional[Dict[str, Any]] = None,
                dedupe: bool = True) -> Tuple[Optional[int], str]:
        text = plainify(text)
        if len(text) < 80:
            return None, "too-short"
        digest = sha256(text)
        if dedupe and self.has_content(digest):
            return None, "exact-duplicate"
        toks = content_words(text)
        if len(toks) < 12:
            return None, "too-few-words"
        sh = simhash64(toks)
        if dedupe:
            nd = self.near_duplicate(sh)
            if nd is not None:
                return None, f"near-duplicate-of:{nd}"
        lang = detect_lang(text)
        cur = self.c.execute(
            """INSERT INTO docs(url,title,source_id,kind,sha256,simhash,lang,bytes,words,
                                fetched_at,status,split,meta)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (url[:1000], (title or "")[:400], source_id, kind, digest, sh, lang,
             len(text.encode("utf-8", "replace")), len(toks), now_iso(), "ok", split,
             json.dumps(meta or {}, ensure_ascii=False)))
        doc_id = int(cur.lastrowid)
        for i, ch in enumerate(chunk_text(text)):
            cd = sha256(f"{doc_id}:{i}:{ch}")
            self.c.execute("INSERT OR IGNORE INTO chunks(doc_id,ord,text,sha256,words) VALUES(?,?,?,?,?)",
                           (doc_id, i, ch, cd, len(words(ch))))
        return doc_id, "ok"

    # --- reads ----------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        q = self.c.execute
        d = q("SELECT COUNT(*) n, COALESCE(SUM(bytes),0) b, COALESCE(SUM(words),0) w FROM docs").fetchone()
        ch = q("SELECT COUNT(*) n, COALESCE(SUM(words),0) w FROM chunks").fetchone()
        per_src = {r["source_id"]: r["n"] for r in q(
            "SELECT source_id, COUNT(*) n FROM docs GROUP BY source_id ORDER BY n DESC LIMIT 12")}
        return {
            "docs": d["n"], "bytes": d["b"], "words": d["w"],
            "chunks": ch["n"], "chunk_words": ch["w"],
            "tokens_approx": int(d["w"] * 1.33),
            "eval_items": q("SELECT COUNT(*) n FROM eval_items").fetchone()["n"],
            "cycles": q("SELECT COALESCE(MAX(cycle),0) n FROM runs").fetchone()["n"],
            "lessons": q("SELECT COUNT(*) n FROM lessons").fetchone()["n"],
            "by_source": per_src,
        }

    def chunk_rows(self, limit: Optional[int] = None, doc_ids: Optional[Iterable[int]] = None):
        sql = "SELECT c.id, c.doc_id, c.text, d.url, d.title, d.split, d.fetched_at FROM chunks c JOIN docs d ON d.id=c.doc_id"
        args: List[Any] = []
        if doc_ids is not None:
            ids = list(doc_ids)
            if not ids:
                return iter([])
            sql += f" WHERE c.doc_id IN ({','.join('?' * len(ids))})"
            args = ids
        sql += " ORDER BY c.id"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        return self.c.execute(sql, args).fetchall()

    def chunk(self, cid: int) -> Optional[sqlite3.Row]:
        return self.c.execute(
            """SELECT c.id,c.doc_id,c.text,d.url,d.title,d.split,d.fetched_at
               FROM chunks c JOIN docs d ON d.id=c.doc_id
               WHERE c.id=?""", (cid,)).fetchone()

    def docs_by_source(self, source_id: str) -> List[sqlite3.Row]:
        return self.c.execute("SELECT * FROM docs WHERE source_id=? ORDER BY id", (source_id,)).fetchall()

    def sources(self) -> List[Dict[str, Any]]:
        rows = self.c.execute(
            """SELECT source_id, COUNT(*) n, COALESCE(SUM(words),0) w, MAX(fetched_at) last
               FROM docs GROUP BY source_id ORDER BY w DESC""").fetchall()
        return [{"id": r["source_id"], "docs": r["n"], "words": r["w"], "last": r["last"]} for r in rows]

    def log_run(self, cycle: int, phase: str, ok: bool, detail: Any, secs: float) -> None:
        self.c.execute("INSERT INTO runs(ts,cycle,phase,ok,detail,secs) VALUES(?,?,?,?,?,?)",
                       (now_iso(), cycle, phase, 1 if ok else 0,
                        json.dumps(detail, ensure_ascii=False, default=str)[:8000], round(secs, 3)))

    def metric(self, cycle: int, name: str, value: float) -> None:
        self.c.execute("INSERT INTO metrics(ts,cycle,name,value) VALUES(?,?,?,?)",
                       (now_iso(), cycle, name, float(value)))

    def metric_history(self, name: str) -> List[Tuple[int, float]]:
        rows = self.c.execute(
            "SELECT cycle, AVG(value) v FROM metrics WHERE name=? GROUP BY cycle ORDER BY cycle",
            (name,)).fetchall()
        return [(int(r["cycle"]), float(r["v"])) for r in rows]

    def leaderboard(self) -> List[Dict[str, Any]]:
        rows = self.c.execute(
            """SELECT cycle, ts, name, AVG(value) v FROM metrics
               WHERE name IN ('score','cloze_acc','qa_f1','retrieval_r1','math_acc',
                              'math_bank_acc','coverage','corpus_docs')
               GROUP BY cycle, name ORDER BY cycle""").fetchall()
        out: Dict[int, Dict[str, Any]] = {}
        for r in rows:
            e = out.setdefault(int(r["cycle"]), {"cycle": int(r["cycle"]), "ts": r["ts"]})
            e[r["name"]] = round(float(r["v"]), 4)
        return [out[k] for k in sorted(out)]


LANG_PAT = [(re.compile(r"\b(the|and|of|to|is|in)\b", re.I), "en"),
            (re.compile(r"\b(der|die|das|und|ist|nicht)\b", re.I), "de"),
            (re.compile(r"\b(le|la|les|est|et|des)\b", re.I), "fr"),
            (re.compile(r"\b(el|la|los|es|que|de)\b", re.I), "es"),
            (re.compile(r"[\u0980-\u09FF]"), "bn"),
            (re.compile(r"[\u0600-\u06FF]"), "ar"),
            (re.compile(r"[\u4E00-\u9FFF]"), "zh")]


def detect_lang(text: str) -> str:
    sample = text[:1500]
    best, bestn = "en", 0
    for rx, code in LANG_PAT:
        n = len(rx.findall(sample))
        if n > bestn:
            best, bestn = code, n
    return best


# ==============================================================================
# 7. CHUNKER
# ==============================================================================
def chunk_text(text: str, target_tokens: int = 180, overlap_tokens: int = 30,
               min_tokens: int = 20) -> List[str]:
    """Paragraph-aware sliding window. Keeps sentences intact; carries an overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 2]
    if not paras:
        paras = [text.strip()]
    units: List[str] = []
    for p in paras:
        if approx_tokens(p) <= target_tokens * 1.4:
            units.append(p)
        else:
            units.extend(sentences(p))
    out: List[str] = []
    cur: List[str] = []
    cur_tok = 0
    for u in units:
        t = approx_tokens(u)
        if cur and cur_tok + t > target_tokens:
            out.append("\n".join(cur).strip())
            # keep a tail for context continuity
            keep: List[str] = []
            keep_tok = 0
            for s in reversed(cur):
                st = approx_tokens(s)
                if keep_tok + st > overlap_tokens:
                    break
                keep.insert(0, s)
                keep_tok += st
            cur, cur_tok = keep, keep_tok
        cur.append(u)
        cur_tok += t
    if cur:
        out.append("\n".join(cur).strip())
    return [c for c in out if approx_tokens(c) >= min_tokens or len(out) == 1]


# ==============================================================================
# 8. EMBEDDINGS
# ==============================================================================
class HashEmbedder:
    """
    Dependency-free dense vectoriser: word uni/bigrams + character 4-grams hashed
    into a fixed vector with sub-linear term frequency, then L2-normalised.

    This is a genuine (if lexical) semantic representation -- the same family as
    the hashing trick / fastText subword vectors. If scikit-learn is installed
    the LSA stage below turns these into true distributional-semantic vectors.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _feats(self, text: str) -> Dict[str, int]:
        w = words(text)
        f: Dict[str, int] = collections.Counter()
        for t in w:
            f["w:" + t] += 1
        for a, b in zip(w, w[1:]):
            f["b:" + a + " " + b] += 1
        s = "".join(w)
        L = len(s)
        if L >= 4:
            s = "\x02" + s + "\x03"
            for i in range(len(s) - 3):
                f["c:" + s[i:i + 4]] += 1
        return f

    def embed(self, text: str) -> "np.ndarray":
        v = np.zeros(self.dim, dtype=np.float32)
        for k, n in self._feats(text).items():
            idx = blake_int(k) % self.dim
            v[idx] += 1.0 + math.log(n)
        nz = float(np.linalg.norm(v))
        if nz > 0:
            v /= nz
        return v

    def embed_many(self, texts: Iterable[str], batch_log: int = 0) -> "np.ndarray":
        rows = [self.embed(t) for t in texts]
        return np.vstack(rows) if rows else np.zeros((0, self.dim), dtype=np.float32)


class SemanticIndex:
    """
    Two-tier representation:
      lexical : BM25 over an inverted index            (exact term evidence)
      dense   : TF-IDF -> truncated SVD (LSA) cosine   (distributional semantics)
    Both are rebuilt incrementally and persisted; the rebuild is skipped whenever
    the corpus signature is unchanged, which is what keeps CPU load near zero.
    """

    def __init__(self, cfg: Dict[str, Any], dirpath: Path):
        self.cfg = cfg
        self.dir = dirpath
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = int(cfg.get("retrieval", {}).get("lsa_dim", 256))
        self.hasher = HashEmbedder(512)
        self.ids: List[int] = []
        self.dense: Optional["np.ndarray"] = None       # float32 (n, dim), L2-normalised
        self.quant: Optional["np.ndarray"] = None       # int8 on disk
        self.vectorizer = None
        self.svd = None
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        self.dlen: Dict[int, int] = {}
        self.avgdl = 1.0
        self.signature: str = ""
        self.built_at = ""

    # --- persistence ----------------------------------------------------------
    def paths(self) -> Dict[str, Path]:
        return {"meta": self.dir / "index.meta.json", "vec": self.dir / "index.vec.npz",
                "lex": self.dir / "index.lex.pkl.gz"}

    def save(self) -> None:
        p = self.paths()
        if self.quant is not None and np is not None:
            np.savez_compressed(p["vec"], q=self.quant, ids=np.array(self.ids, dtype=np.int64))
        with gzip.open(p["lex"], "wb", compresslevel=6) as f:
            pickle.dump({"postings": self.postings, "dlen": self.dlen, "avgdl": self.avgdl}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        obj = None
        if self.vectorizer is not None:
            try:
                obj = pickle.dumps({"vectorizer": self.vectorizer, "svd": self.svd})
            except Exception:
                obj = None
        if obj is not None:
            (self.dir / "index.lsa.pkl.gz").write_bytes(gzip.compress(obj, 6))
        p["meta"].write_text(json.dumps({
            "signature": self.signature, "built_at": self.built_at, "dim": self.dim,
            "n": len(self.ids), "backend": "lsa+bm25" if SKLEARN else "hash+bm25",
            "vocab": int(getattr(self.vectorizer, "vocabulary_", {}) and len(self.vectorizer.vocabulary_) or 0),
        }, indent=2))

    def load(self) -> bool:
        p = self.paths()
        if not (p["meta"].exists() and p["lex"].exists()):
            return False
        try:
            meta = json.loads(p["meta"].read_text())
            with gzip.open(p["lex"], "rb") as f:
                lex = pickle.load(f)
            self.postings, self.dlen, self.avgdl = lex["postings"], lex["dlen"], lex["avgdl"]
            if p["vec"].exists() and np is not None:
                z = np.load(p["vec"])
                self.quant = z["q"]
                self.ids = [int(x) for x in z["ids"]]
                self.dense = (self.quant.astype(np.float32) / 127.0)
            lsa = self.dir / "index.lsa.pkl.gz"
            if lsa.exists():
                try:
                    o = pickle.loads(gzip.decompress(lsa.read_bytes()))
                    self.vectorizer, self.svd = o["vectorizer"], o["svd"]
                except Exception:
                    self.vectorizer = self.svd = None
            self.signature = meta.get("signature", "")
            self.built_at = meta.get("built_at", "")
            self.dim = meta.get("dim", self.dim)
            return True
        except Exception as e:
            log.w(f"index load failed ({e}); will rebuild")
            return False

    # --- build ----------------------------------------------------------------
    def build(self, store: "Store", force: bool = False) -> Dict[str, Any]:
        rows = store.chunk_rows()
        sig = f"{len(rows)}:{rows[-1]['id'] if rows else 0}"
        if not force and self.load() and self.signature == sig:
            return {"rebuilt": False, "n": len(self.ids), "signature": sig}
        if not rows:
            self.ids, self.signature, self.built_at = [], sig, now_iso()
            self.dense = np.zeros((0, self.dim), dtype=np.float32) if np is not None else None
            self.quant = None
            self.postings, self.dlen, self.avgdl = {}, {}, 1.0
            self.save()
            return {"rebuilt": True, "n": 0, "signature": sig}

        self.ids = [int(r["id"]) for r in rows]
        texts = [r["text"] for r in rows]
        tok_cache = [content_words(t) for t in texts]

        # lexical: inverted index token -> [(chunk_id, tf)]
        postings: Dict[str, List[Tuple[int, int]]] = {}
        dlen: Dict[int, int] = {}
        for cid, toks in zip(self.ids, tok_cache):
            dlen[cid] = max(1, len(toks))
            tf = collections.Counter(t for t in toks if t not in STOPWORDS and len(t) > 1)
            for tok, n in tf.items():
                postings.setdefault(tok, []).append((cid, n))
        # prune ultra-rare terms only if the index is getting large
        if len(postings) > 400_000:
            postings = {k: v for k, v in postings.items() if len(v) > 1 or len(k) > 3}
        self.postings, self.dlen = postings, dlen
        self.avgdl = float(sum(dlen.values()) / max(1, len(dlen)))

        # dense
        if SKLEARN and len(rows) >= 12:
            n_comp = int(clamp(self.dim, 8, min(300, max(8, len(rows) - 1))))
            self.vectorizer = TfidfVectorizer(sublinear_tf=True, lowercase=True,
                                              max_features=60_000, min_df=1,
                                              ngram_range=(1, 2), token_pattern=r"[a-z][a-z'\-]+|\d+")
            X = self.vectorizer.fit_transform(texts)
            self.svd = TruncatedSVD(n_components=n_comp, random_state=0)
            M = self.svd.fit_transform(X)
            M = np.nan_to_num(M).astype(np.float32)
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            M /= norms
            self.dense = M
            self.dim = n_comp
        else:
            self.dense = self.hasher.embed_many(texts)
            self.dim = self.dense.shape[1] if self.dense.size else 512

        self.quant = np.clip(np.round(self.dense * 127.0), -127, 127).astype(np.int8)
        self.signature, self.built_at = sig, now_iso()
        self.save()
        return {"rebuilt": True, "n": len(self.ids), "signature": sig,
                "vocab": len(self.postings), "dim": self.dim,
                "backend": "lsa+bm25" if SKLEARN else "hash+bm25"}

    # --- query ----------------------------------------------------------------
    def dense_query(self, q: str) -> Optional["np.ndarray"]:
        if self.dense is None or not len(self.ids):
            return None
        if self.vectorizer is not None and self.svd is not None:
            try:
                v = self.svd.transform(self.vectorizer.transform([q])).astype(np.float32)[0]
                n = float(np.linalg.norm(v))
                return v / n if n else None
            except Exception:
                pass
        v = self.hasher.embed(q)
        if v.shape[0] != self.dense.shape[1]:
            return None
        return v

    def bm25(self, q: str, k1: float, b: float, pool: int) -> List[Tuple[int, float]]:
        toks = [t for t in content_words(q) if t in self.postings]
        if not toks:
            return []
        N = max(1, len(self.ids))
        acc: Dict[int, float] = collections.defaultdict(float)
        for t in toks:
            pl = self.postings[t]
            df = len(pl)
            idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
            for cid, tf in pl:
                dl = self.dlen.get(cid, 1)
                acc[cid] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avgdl))
        top = sorted(acc.items(), key=lambda kv: -kv[1])[:pool]
        if not top:
            return []
        mx = top[0][1] or 1.0
        return [(cid, s / mx) for cid, s in top]

    def dense_scores(self, qv: "np.ndarray") -> List[Tuple[int, float]]:
        sims = self.dense @ qv                                     # (n,)
        order = np.argsort(-sims)[:2000]
        return [(self.ids[i], float(sims[i])) for i in order if sims[i] > 0]

# ==============================================================================
# 9. RETRIEVER -- hybrid lexical + dense, RRF fusion, then a cheap rerank
# ==============================================================================
class Hit(Dict[str, Any]):
    pass


class Retriever:
    def __init__(self, store: Store, index: SemanticIndex, cfg: Dict[str, Any]):
        self.store, self.index, self.cfg = store, index, cfg
        self._meta: Dict[int, sqlite3.Row] = {}

    def _row(self, cid: int) -> Optional[sqlite3.Row]:
        r = self._meta.get(cid)
        if r is None:
            r = self.store.chunk(cid)
            if r is not None:
                self._meta[cid] = r
        return r

    def idf(self, tok: str) -> float:
        pl = self.index.postings.get(tok)
        N = max(1, len(self.index.ids))
        df = len(pl) if pl else 0
        return math.log(1.0 + (N - df + 0.5) / (df + 0.5))

    def search(self, q: str, k: Optional[int] = None, exclude: Iterable[int] = (),
               overrides: Optional[Dict[str, Any]] = None) -> List[Hit]:
        R = dict(self.cfg.get("retrieval", {}))
        if overrides:
            R.update(overrides)
        k = int(k or R.get("top_k", 6))
        pool = int(R.get("candidate_pool", 80))
        excl = set(exclude or ())
        if not self.index.ids:
            return []

        lex = self.index.bm25(q, float(R.get("bm25_k1", 1.5)), float(R.get("bm25_b", 0.75)), pool * 2)
        qv = self.index.dense_query(q)
        den = self.index.dense_scores(qv) if qv is not None else []
        if excl:
            lex = [(c, s) for c, s in lex if c not in excl]
            den = [(c, s) for c, s in den if c not in excl]

        rrf_k = float(R.get("rrf_k", 60))
        wl, wd = float(R.get("lexical_weight", 0.45)), float(R.get("dense_weight", 0.55))
        fused: Dict[int, float] = collections.defaultdict(float)
        lexmap = {c: s for c, s in lex}
        denmap = {c: s for c, s in den}
        for rank, (cid, _) in enumerate(lex[:pool]):
            fused[cid] += wl / (rrf_k + rank + 1)
        for rank, (cid, _) in enumerate(den[:pool]):
            fused[cid] += wd / (rrf_k + rank + 1)
        if not fused:
            return []
        mx = max(fused.values()) or 1.0
        cands = sorted(fused.items(), key=lambda kv: -kv[1])[: int(pool * 1.5)]

        qset = set(content_words(q))
        now_ts = time.time()
        out: List[Hit] = []
        for cid, sc in cands:
            row = self._row(cid)
            if row is None:
                continue
            txt = row["text"]
            body = set(content_words(txt))
            cover = len(qset & body) / max(1, len(qset))
            s = sc / mx
            if R.get("rerank", True):
                s += 0.30 * cover
                title = (row["title"] or "").lower()
                if any(w in title for w in qset):
                    s += float(R.get("title_boost", 0.12))
                fa = row["fetched_at"] or ""
                try:
                    age_d = (now_ts - datetime.strptime(fa, "%Y-%m-%dT%H:%M:%SZ").replace(
                        tzinfo=timezone.utc).timestamp()) / 86400.0
                    s += float(R.get("freshness_boost", 0.03)) * clamp(1.0 - age_d / 30.0, 0.0, 1.0)
                except Exception:
                    pass
            out.append(Hit(chunk_id=cid, doc_id=int(row["doc_id"]), text=txt,
                           url=row["url"], title=row["title"], score=round(float(s), 5),
                           lexical=round(float(lexmap.get(cid, 0.0)), 5),
                           dense=round(float(denmap.get(cid, 0.0)), 5), coverage=round(cover, 4)))
        out.sort(key=lambda h: -h["score"])
        # collapse near-identical neighbours so one document cannot flood the answer
        ded: List[Hit] = []
        seen: List[set] = []
        for h in out:
            body = set(content_words(h["text"]))
            if any(len(body & s) / max(1, min(len(body), len(s))) > 0.72 for s in seen):
                continue
            ded.append(h)
            seen.append(body)
            if len(ded) >= k:
                break
        return ded
