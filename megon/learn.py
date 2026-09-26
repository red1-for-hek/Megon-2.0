"""MEGON I AI -- self-tuning bandit, acquisition, the learn cycle, SFT export"""
from __future__ import annotations
from megon.base import *
from megon.memory import *
from megon.mind import *
from megon.evals import *

# ==============================================================================
# 14. SELF-TUNER -- UCB1 bandit over MEGON's own hyper-parameters
# ==============================================================================
ARMS: List[Tuple[str, Dict[str, Any]]] = [
    ("base", {}),
    ("dense_heavy", {"dense_weight": 0.78, "lexical_weight": 0.22}),
    ("lexical_heavy", {"dense_weight": 0.22, "lexical_weight": 0.78}),
    ("wide_recall", {"top_k": 10, "candidate_pool": 140}),
    ("tight_precision", {"top_k": 4, "candidate_pool": 48}),
    ("bm25_long_docs", {"bm25_k1": 1.2, "bm25_b": 0.5}),
    ("bm25_short_docs", {"bm25_k1": 2.0, "bm25_b": 0.9}),
    ("rrf_sharp", {"rrf_k": 20}),
    ("rrf_soft", {"rrf_k": 100}),
    ("no_rerank", {"rerank": False}),
    ("title_matters", {"title_boost": 0.25, "freshness_boost": 0.0}),
]


class Tuner:
    def __init__(self, c: sqlite3.Connection):
        self.c = c
        st = kv_get(c, "bandit", {}) or {}
        self.state: Dict[str, Dict[str, float]] = {
            name: {"n": float(st.get(name, {}).get("n", 0)),
                   "mean": float(st.get(name, {}).get("mean", 0.0))}
            for name, _ in ARMS
        }

    def save(self) -> None:
        kv_set(self.c, "bandit", self.state)

    def ucb_select(self, n_arms: int = 3) -> List[Tuple[str, Dict[str, Any]]]:
        total = sum(s["n"] for s in self.state.values()) or 1.0
        scored = []
        for name, ov in ARMS:
            s = self.state[name]
            if s["n"] == 0:
                bonus = 10.0
            else:
                bonus = math.sqrt(2.0 * math.log(total) / s["n"])
            scored.append((s["mean"] + bonus, name, ov))
        scored.sort(key=lambda t: -t[0])
        return [(n, o) for _, n, o in scored[:n_arms]]

    def update(self, name: str, reward: float) -> None:
        s = self.state.setdefault(name, {"n": 0.0, "mean": 0.0})
        s["n"] += 1
        s["mean"] += (reward - s["mean"]) / s["n"]
        self.save()

    def best(self) -> Tuple[str, Dict[str, Any]]:
        tried = [(s["mean"], s["n"], name, ov) for name, ov in ARMS for s in [self.state[name]] if s["n"] > 0]
        if not tried:
            return "base", {}
        tried.sort(key=lambda t: -t[0])
        return tried[0][2], dict(tried[0][3])

    def table(self) -> List[Dict[str, Any]]:
        rows = [{"arm": n, "n": int(s["n"]), "mean": round(s["mean"], 4)}
                for n, s in self.state.items()]
        rows.sort(key=lambda r: -r["mean"])
        return rows


# ==============================================================================
# 15. ACQUISITION -- slow drip from real public corpora
# ==============================================================================
HF_ROWS = "https://datasets-server.huggingface.co/rows"
WIKI_API = "https://en.wikipedia.org/w/api.php"


def hf_pull(src: Dict[str, Any], cfg: Dict[str, Any], rate: RateLimiter,
            policy: Policy, c: sqlite3.Connection, length: int = 50,
            offset: Optional[int] = None) -> Tuple[List[Dict[str, Any]], int, str]:
    """One paged pull. Returns (rows, start_offset, status).

    The caller advances the cursor by however many rows it actually *consumed*,
    so a budget cut-off in the middle of a page does not silently skip data.
    """
    off = offset if offset is not None else int(kv_get(c, f"hf_offset:{src['id']}", 0))
    params = {"dataset": src["dataset"], "config": src.get("config", "default"),
              "split": src.get("split", "train"), "offset": off, "length": length}
    url = HF_ROWS + "?" + urllib.parse.urlencode(params)
    data, status = http_get(url, cfg, rate, policy, c, max_bytes=6_000_000)
    if data is None:
        return [], off, status
    try:
        d = json.loads(data.decode("utf-8", "replace"))
    except Exception as e:
        return [], off, f"parse:{e}"
    rows = [r.get("row", {}) for r in d.get("rows", [])]
    return rows, off, "ok"


def hf_advance(c: sqlite3.Connection, src_id: str, start: int, consumed: int) -> None:
    kv_set(c, f"hf_offset:{src_id}", start + consumed)


def ingest_source(src: Dict[str, Any], cfg: Dict[str, Any], store: Store, budget: Budget,
                  rate: RateLimiter, policy: Policy, eval_every: int = 5,
                  slice_docs: int = 12) -> Dict[str, Any]:
    """Ingest one *slice* of one source. Always bounded by `budget`.

    `slice_docs` keeps any single corpus from swallowing the whole daily budget,
    which is what gives broad coverage instead of 40 rows of one dataset.
    """
    c = store.c
    res: Dict[str, Any] = {"source": src["id"], "docs": 0, "skipped": collections.Counter(),
                           "eval_items": 0, "bytes": 0}
    if budget.exhausted():
        res["skipped"]["budget"] += 1
        return res

    if src.get("kind") == "hf":
        want = max(5, min(int(slice_docs), budget.max_docs - budget.docs + 5))
        rows, off, status = hf_pull(src, cfg, rate, policy, c, length=min(60, want))
        res["offset"] = off
        res["status"] = status
        if not rows:
            return res
        consumed = 0
        tf = src.get("text_field", "text")
        for i, row in enumerate(rows):
            if budget.exhausted() or res["docs"] + res["eval_items"] >= slice_docs:
                break
            consumed = i
            # ---- sources that carry their own labels become benchmark items ----
            if src.get("qa") and row.get("question"):
                ans = row.get("answers") or {}
                texts = ans.get("text") if isinstance(ans, dict) else None
                gold = (texts[0] if texts else None) or str(ans)
                ctx = row.get("context") or ""
                if ctx and gold:
                    nb = len(ctx.encode("utf-8", "replace"))
                    res["bytes"] += nb
                    if not policy.check_content(ctx, src["dataset"], c):
                        budget.charge(nb)
                        continue
                    if not src.get("eval_only"):
                        doc_id, why = store.add_doc(src["dataset"] + "#ctx", str(row.get("title", "")),
                                                    ctx, src["id"], "hf", split="train",
                                                    meta={"dataset": src["dataset"]})
                        if doc_id is None:
                            res["skipped"][why] += 1
                    eid = c.execute("SELECT id FROM chunks WHERE doc_id=(SELECT id FROM docs WHERE sha256=? LIMIT 1) ORDER BY ord LIMIT 1",
                                    (sha256(plainify(ctx)),)).fetchone()
                    c.execute("""INSERT INTO eval_items(kind,question,answer,options,evidence_chunk,
                                 source_doc,split,created_at,meta) VALUES(?,?,?,?,?,?,?,?,?)""",
                              ("qa", row["question"], gold, None,
                               eid["id"] if eid else None, None, "eval", now_iso(),
                               json.dumps({"dataset": src["dataset"], "real": True})))
                    res["eval_items"] += 1
                    budget.charge(nb)
                continue
            if src.get("math") and row.get("question"):
                gold = final_answer_of(str(row.get("answer", "")))
                if gold is None:
                    continue
                if not policy.check_content(str(row["question"]), src["dataset"], c):
                    continue
                c.execute("""INSERT INTO eval_items(kind,question,answer,options,evidence_chunk,
                             source_doc,split,created_at,meta) VALUES(?,?,?,?,?,?,?,?,?)""",
                          ("math", row["question"], str(int(gold)), None, None, None, "eval",
                           now_iso(), json.dumps({"dataset": src["dataset"], "real": True})))
                res["eval_items"] += 1
                continue
            text = row.get(tf) or ""
            if not isinstance(text, str) or len(text) < 120:
                res["skipped"]["short"] += 1
                continue
            if src.get("instruct"):
                ins = row.get("instruction") or row.get("prompt") or ""
                text = (ins + "\n\n" + text).strip()
            text = text[: int(cfg["budget"]["max_bytes_per_doc"])]
            res["bytes"] += len(text.encode("utf-8", "replace"))
            if not policy.check_content(text, src["dataset"], c):
                res["skipped"]["policy"] += 1
                continue
            rid = str(row.get("id") or row.get("url") or i)
            url = row.get("url") or f"hf://{src['dataset']}/{rid}"
            title = (str(row.get("title")) if row.get("title") else text[:70]).strip()
            split = "eval" if (i % eval_every == 0) else "train"
            doc_id, why = store.add_doc(url, title, text, src["id"], "hf", split=split,
                                        meta={"dataset": src["dataset"], "row": rid})
            if doc_id:
                res["docs"] += 1
            else:
                res["skipped"][why] += 1
            budget.charge(len(text.encode("utf-8", "replace")))
        hf_advance(c, src["id"], off, max(consumed, min(len(rows), consumed + 1)))
        res["next_offset"] = int(kv_get(c, f"hf_offset:{src['id']}", 0))
        return res

    # ---- plain URL source ----
    url = src.get("url", "")
    data, status = http_get(url, cfg, rate, policy, c)
    res["status"] = status
    if data is None:
        return res
    res["bytes"] += len(data)
    ctype = "html"
    if url.endswith((".txt", ".md")):
        text, title, meta = data.decode("utf-8", "replace"), url.rsplit("/", 1)[-1], {}
    else:
        try:
            text, title, meta = html_to_text(data, url)
        except Exception as e:
            res["skipped"][f"extract:{e}"] += 1
            return res
        ctype = "html"
    if not policy.check_content(text, url, c):
        res["skipped"]["policy"] += 1
        return res
    split = "eval" if (store.stats()["docs"] % eval_every == 0) else "train"
    doc_id, why = store.add_doc(url, title, text, src["id"], "url", split=split,
                                meta={"ctype": ctype, **meta})
    if doc_id:
        res["docs"] += 1
    else:
        res["skipped"][why] += 1
    budget.charge(len(data))
    return res


def wikipedia_lookup(topic: str, cfg: Dict[str, Any], store: Store, budget: Budget,
                     rate: RateLimiter, policy: Policy) -> int:
    """Closed loop: a topic flagged weak by the evaluator gets targeted acquisition."""
    c = store.c
    q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": topic,
                                "srlimit": 3, "format": "json"})
    data, status = http_get(WIKI_API + "?" + q, cfg, rate, policy, c, max_bytes=200_000)
    if data is None:
        return 0
    try:
        hits = json.loads(data.decode("utf-8", "replace"))["query"]["search"]
    except Exception:
        return 0
    n = 0
    for h in hits:
        if budget.exhausted():
            break
        title = h["title"]
        qq = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1,
                                     "format": "json", "titles": title, "redirects": 1})
        d2, st = http_get(WIKI_API + "?" + qq, cfg, rate, policy, c, max_bytes=1_200_000)
        if d2 is None:
            continue
        try:
            pages = json.loads(d2.decode("utf-8", "replace"))["query"]["pages"]
            page = list(pages.values())[0]
            txt = page.get("extract") or ""
        except Exception:
            continue
        if len(txt) < 200 or not policy.check_content(txt, title, c):
            continue
        doc_id, why = store.add_doc(f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                                    title, txt, "wikipedia-targeted", "wiki", split="train",
                                    meta={"topic": topic})
        if doc_id:
            n += 1
            budget.charge(len(txt.encode("utf-8", "replace")))
    return n


# ==============================================================================
# 16. THE LEARN CYCLE -- acquire -> consolidate -> evaluate -> tune -> reflect
# ==============================================================================
def next_cycle(c: sqlite3.Connection) -> int:
    n = int(kv_get(c, "cycle", 0)) + 1
    kv_set(c, "cycle", n)
    return n


def learn_cycle(cfg: Dict[str, Any], mb: Optional[float] = None, docs: Optional[int] = None,
                minutes: Optional[float] = None, verbose: bool = False,
                skip_tune: bool = False, n_arms: int = 3) -> Dict[str, Any]:
    c = init_db()
    store = Store(c)
    index = SemanticIndex(cfg, home_dir() / "index")
    retriever = Retriever(store, index, cfg)
    solver_w = kv_get(c, "solver_weights", None)
    solver = MathSolver(solver_w)
    answerer = Answerer(store, retriever, cfg, solver)
    evaluator = Evaluator(store, retriever, answerer, cfg)
    tuner = Tuner(c)
    cyc = next_cycle(c)
    B = cfg.get("budget", {})
    budget = Budget(mb if mb is not None else B.get("mb_per_day", 6.0),
                    docs if docs is not None else B.get("max_docs_per_day", 40),
                    minutes if minutes is not None else B.get("minutes_per_cycle", 20), cfg)
    rate = RateLimiter(B.get("min_request_interval_s", 1.5), B.get("max_requests_per_min", 40))
    policy = Policy(cfg)
    report: Dict[str, Any] = {"cycle": cyc, "started": now_iso(), "phases": {}}
    log.i(f"=== MEGON learn cycle #{cyc} :: budget {budget.max_bytes/1048576:.1f} MB / "
          f"{budget.max_docs} docs ===")

    # -- 1. ACQUIRE ----------------------------------------------------------
    t = time.time()
    acq: List[Dict[str, Any]] = []
    sources = sorted(cfg.get("sources", []), key=lambda s: -float(s.get("priority", 0.5)))
    wishlist = kv_get(c, "wishlist", []) or []
    log.i(f"acquire: {len(sources)} sources, {len(wishlist)} queued weak topics")
    # Round-robin: each corpus gets a small slice per pass, so a cycle spreads
    # across the whole source list instead of draining the first one.
    per_slice = max(3, min(12, budget.max_docs // max(1, len(sources)) or 3))
    taken: collections.Counter = collections.Counter()
    cap_each = max(per_slice * 3, budget.max_docs // max(1, len(sources)) + 2)
    for rnd in range(6):
        progressed = False
        for src in sources:
            if budget.exhausted():
                break
            if taken[src["id"]] >= cap_each:
                continue
            try:
                r = ingest_source(src, cfg, store, budget, rate, policy, slice_docs=per_slice)
            except Exception as e:
                r = {"source": src.get("id"), "error": f"{type(e).__name__}: {e}"}
                log.w(f"source {src.get('id')} failed: {e}")
            r["skipped"] = dict(r.get("skipped", {})) if isinstance(r.get("skipped"), collections.Counter) \
                else r.get("skipped", {})
            got = int(r.get("docs", 0)) + int(r.get("eval_items", 0))
            taken[src["id"]] += got
            if got:
                progressed = True
                acq.append(r)
                log.i(f"  + {src['id']}: {r.get('docs',0)} docs, {r.get('eval_items',0)} eval items, "
                      f"{r.get('bytes',0)/1024:.0f} KB")
            elif r.get("error") or (r.get("status", "ok") not in ("ok", "ok:truncated")):
                acq.append(r)
            if verbose:
                log.d(f"    {json.dumps(r, ensure_ascii=False, default=str)[:300]}")
        if budget.exhausted() or not progressed:
            break
    targeted = 0
    for topic in wishlist[:6]:
        if budget.exhausted():
            break
        try:
            n = wikipedia_lookup(str(topic), cfg, store, budget, rate, policy)
            targeted += n
            if n:
                log.i(f"  + targeted '{topic}': {n} docs (weak-topic feedback loop)")
        except Exception as e:
            log.w(f"targeted {topic} failed: {e}")
    report["phases"]["acquire"] = {"secs": round(time.time() - t, 2), "results": acq,
                                   "targeted_docs": targeted, "budget": budget.summary(),
                                   "politeness_sleep_s": round(rate.slept, 1)}
    store.log_run(cyc, "acquire", True, {"docs": budget.docs, "mb": round(budget.bytes/1048576, 3),
                                         "stops": budget.summary()["stops"]}, time.time() - t)

    # -- 2. CONSOLIDATE ------------------------------------------------------
    t = time.time()
    built = index.build(store, force=True)
    report["phases"]["consolidate"] = {"secs": round(time.time() - t, 2), **built}
    log.i(f"consolidate: {built.get('n',0)} chunks indexed via {built.get('backend','hash+bm25')} "
          f"in {time.time()-t:.1f}s")
    store.log_run(cyc, "consolidate", True, built, time.time() - t)

    # -- 3. EVALUATE (baseline = current best config) ------------------------
    t = time.time()
    gen = evaluator.generate_items(max_items=45)
    best_arm, best_ov = tuner.best()
    base = evaluator.run(overrides=best_ov)
    report["phases"]["evaluate"] = {"secs": round(time.time() - t, 2), "generated": gen,
                                    "baseline": base, "baseline_arm": best_arm}
    log.i(f"evaluate: {base.get('n',0)} items, {sum(gen.values())} new :: score={base.get('score',0):.3f} "
          f"cloze={base.get('cloze_acc',0):.2f} qa_f1={base.get('qa_f1',0):.2f} "
          f"retr={base.get('retrieval_r1',0):.2f} math={base.get('math_acc',0):.2f}")
    store.log_run(cyc, "evaluate", True, {"score": base.get("score"), "n": base.get("n")}, time.time() - t)

    # -- 4. SELF-TUNE (bandit) ----------------------------------------------
    t = time.time()
    tune_log: List[Dict[str, Any]] = []
    final_score = base.get("score", 0.0)
    if not skip_tune and base.get("n", 0) >= 4:
        for arm, ov in tuner.ucb_select(n_arms):
            if budget.deadline and time.time() > budget.deadline - 15:
                break
            r = evaluator.run(overrides=ov, limit=120)
            tuner.update(arm, float(r.get("score", 0.0)))
            tune_log.append({"arm": arm, "score": round(float(r.get("score", 0.0)), 4), "ov": ov})
            log.i(f"  tune arm {arm:16s} -> {r.get('score',0):.4f}")
        best_arm, best_ov = tuner.best()
        final = evaluator.run(overrides=best_ov, limit=200)
        final_score = float(final.get("score", 0.0))
        report["phases"]["tune"] = {"secs": round(time.time() - t, 2), "tried": tune_log,
                                    "winner": best_arm, "winner_overrides": best_ov,
                                    "final": final}
        base = final
        log.ok(f"self-tune: winner '{best_arm}' score={final_score:.4f}")
    kv_set(c, "best_arm", best_arm)
    kv_set(c, "best_overrides", best_ov)
    store.log_run(cyc, "tune", True, {"winner": best_arm, "score": final_score}, time.time() - t)

    # -- 5. REFLECT ----------------------------------------------------------
    t = time.time()
    lessons: List[Dict[str, Any]] = []

    def add_lesson(kind: str, text: str, weight: float = 1.0) -> None:
        c.execute("INSERT INTO lessons(ts,cycle,kind,text,weight) VALUES(?,?,?,?,?)",
                  (now_iso(), cyc, kind, text, weight))
        lessons.append({"kind": kind, "text": text, "weight": weight})

    # 5a. learn the math strategy priors from this cycle's graded results
    sw = dict(solver.weights)
    for strat, win in (base.get("strategy_win") or {}).items():
        n = int(base.get("strategy_n", {}).get(strat, 0))
        if n >= 3:
            sw[strat] = clamp(sw.get(strat, 1.0) * (0.75 + 0.4 * win), 0.30, 4.0)
            add_lesson("skill", f"math strategy '{strat}' win-rate {win:.2f} over {n} items "
                                f"-> prior now {sw[strat]:.2f}", win)
    kv_set(c, "solver_weights", sw)
    answerer.solver.weights = sw

    # 5b. weak topics become next cycle's crawl queue (the actual feedback loop)
    wl = kv_get(c, "wishlist", []) or []
    for topic, n in (base.get("weak_topics") or [])[:5]:
        topic = topic.strip()
        if len(topic) > 6 and topic.lower() not in {w.lower() for w in wl}:
            wl.append(topic)
            add_lesson("gap", f"weak on '{topic}' ({n} failed items) -> queued for targeted crawl", 1.0)
    kv_set(c, "wishlist", wl[-40:])

    # 5c. which sources actually produced usable evidence?
    # 5c. which sources actually produced usable evidence?
    for r in c.execute("""SELECT d.source_id, COUNT(*) n FROM eval_results e
                          JOIN eval_items i ON i.id=e.item_id
                          JOIN docs d ON d.id=i.source_doc
                          WHERE e.cycle=? AND e.score>=0.7 GROUP BY d.source_id
                          ORDER BY n DESC LIMIT 5""", (cyc,)).fetchall():
        add_lesson("yield", f"source '{r['source_id']}' produced {r['n']} high-scoring answers", 1.0)

    st = store.stats()
    prev = store.leaderboard()
    prev_score = None
    for row in prev:
        if row["cycle"] == cyc - 1:
            prev_score = row.get("score")
    delta = (final_score - prev_score) if prev_score is not None else None
    add_lesson("status", f"cycle {cyc}: score {final_score:.3f}"
                         + (f" ({'+' if (delta or 0) >= 0 else ''}{delta:.3f} vs last cycle)"
                            if delta is not None else " (first measured cycle)"), 0.5)
    report["phases"]["reflect"] = {"secs": round(time.time() - t, 2), "lessons": lessons,
                                   "wishlist": len(wl)}
    store.log_run(cyc, "reflect", True, {"lessons": len(lessons)}, time.time() - t)

    # -- 6. PUBLISH ----------------------------------------------------------
    for name in ("score", "cloze_acc", "qa_f1", "qa_em", "retrieval_r1", "retrieval_r3",
                 "math_acc", "math_bank_acc", "coverage"):
        if name in base:
            store.metric(cyc, name, float(base[name]))
    store.metric(cyc, "corpus_docs", float(st["docs"]))
    store.metric(cyc, "corpus_tokens", float(st["tokens_approx"]))
    store.metric(cyc, "eval_items", float(st["eval_items"]))
    # real answer latency, measured (not guessed) on live queries
    probe = [r["question"] for r in c.execute(
        "SELECT question FROM eval_items ORDER BY RANDOM() LIMIT 5").fetchall()] or ["what is this about?"]
    t = time.time()
    for pq in probe:
        answerer.answer(pq, overrides=best_ov)
    store.metric(cyc, "latency_ms", (time.time() - t) * 1000.0 / max(1, len(probe)))
    card = version_card(cyc, cfg, st, base, tuner.table(), lessons, budget, report, delta)
    vdir = home_dir() / "versions"
    vdir.mkdir(exist_ok=True)
    card_path = vdir / f"cycle-{cyc:04d}.md"
    card_path.write_text(card, encoding="utf-8")
    kv_set(c, "last_report", report)
    kv_set(c, "last_cycle_at", now_iso())
    report["phases"]["publish"] = {"card": str(card_path)}
    log.ok(f"published {card_path.name} :: score {final_score:.4f}"
           + (f" ({'+' if (delta or 0) >= 0 else ''}{delta:.3f})" if delta is not None else ""))
    return report


def version_card(cyc: int, cfg: Dict[str, Any], st: Dict[str, Any], ev: Dict[str, Any],
                 arms: List[Dict[str, Any]], lessons: List[Dict[str, Any]], budget: Budget,
                 report: Dict[str, Any], delta: Optional[float]) -> str:
    L: List[str] = []
    A = L.append
    A(f"# MEGON I AI :: cycle {cyc:04d}")
    A("")
    A(f"- **timestamp**: {now_iso()}")
    A(f"- **engine**: v{VERSION}  |  **schema**: v{SCHEMA_VERSION}")
    A(f"- **composite score**: **{ev.get('score', 0):.4f}**"
      + (f"  ({'+' if (delta or 0) >= 0 else ''}{delta:.4f} vs previous cycle)" if delta is not None else "  (first measured cycle)"))
    A("")
    A("## benchmark (self-growing)")
    A("")
    A("| metric | value |")
    A("|---|---|")
    for k in ("n", "score", "cloze_acc", "qa_f1", "qa_em", "retrieval_r1", "retrieval_r3",
              "math_acc", "math_bank_acc", "coverage"):
        if k in ev:
            A(f"| {k} | {ev[k]} |")
    A("")
    A("## memory")
    A("")
    A(f"- documents: **{st['docs']}** ({human(st['bytes'])} on disk)")
    A(f"- chunks: **{st['chunks']}**  |  words: **{human(st['words'], '')}**  |  "
      f"~tokens: **{human(st['tokens_approx'], '')}**")
    A(f"- benchmark items: **{st['eval_items']}**  |  lessons recorded: **{st['lessons']}**")
    A("- by source: " + ", ".join(f"`{k}`={v}" for k, v in list(st["by_source"].items())[:8]))
    A("")
    A("## self-tuning (UCB1 bandit over its own hyper-parameters)")
    A("")
    A("| arm | pulls | mean score |")
    A("|---|---|---|")
    for a in arms[:8]:
        A(f"| {a['arm']} | {a['n']} | {a['mean']} |")
    A("")
    A("## lessons learned this cycle")
    A("")
    for ls in lessons[:14]:
        A(f"- **[{ls['kind']}]** {ls['text']}")
    if not lessons:
        A("- (none)")
    A("")
    A("## hardware cost of this cycle")
    A("")
    bs = budget.summary()
    A(f"- downloaded: **{bs['mb']} MB** across **{bs['docs']}** documents")
    A(f"- stopped by: {', '.join(bs['stops']) or 'finished all sources'}")
    A(f"- politeness sleeps: {report['phases'].get('acquire', {}).get('politeness_sleep_s', 0)} s "
      f"(rate limiter, not CPU)")
    ph = report.get("phases", {})
    A("- phase seconds: " + ", ".join(f"{k}={v.get('secs')}" for k, v in ph.items() if isinstance(v, dict) and "secs" in v))
    A("")
    A("## honesty note")
    A("")
    A("This card measures MEGON against **its own** growing benchmark. A rising curve here means "
      "its retrieval, coverage and arithmetic strategies are measurably improving. It does **not** "
      "mean MEGON beat any frontier model on any public leaderboard, and no amount of crawling "
      "would make it. The route from this to real weight updates is `megon export-sft` + "
      "`finetune-script` on a GPU machine.")
    A("")
    return "\n".join(L)


# ==============================================================================
# 17. EXPORT -> real fine-tuning data (the honest bridge to trained weights)
# ==============================================================================
def export_sft(cfg: Dict[str, Any], limit: int = 5000) -> Path:
    c = init_db()
    store = Store(c)
    index = SemanticIndex(cfg, home_dir() / "index")
    index.load()
    retriever = Retriever(store, index, cfg)
    answerer = Answerer(store, retriever, cfg, MathSolver(kv_get(c, "solver_weights", None)))
    out_dir = home_dir() / "export"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"megon-sft-{today_key()}.jsonl"
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for it in c.execute("SELECT * FROM eval_items WHERE kind IN ('qa','math') ORDER BY id LIMIT ?",
                            (limit,)).fetchall():
            if it["kind"] == "qa":
                ans = answerer.answer(it["question"], k=4)
                target = it["answer"]
                ctx = "\n".join(h["text"][:400] for h in ans["evidence"][:3])
                rec = {"instruction": it["question"],
                       "input": ctx[:2000],
                       "output": target,
                       "source": "mythos-memory/qa",
                       "grounded": bool(ans["evidence"])}
            else:
                sol = answerer.solver.solve(it["question"])
                rec = {"instruction": it["question"], "input": "",
                       "output": f"{sol.get('expr') or ''}\nThe answer is {it['answer']}.",
                       "source": "mythos-memory/math", "grounded": True}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
        # instruction pairs harvested from instruction datasets already in memory
        for d in c.execute("SELECT url,title,meta FROM docs WHERE source_id='alpaca' LIMIT ?",
                           (limit,)).fetchall():
            pass
    log.ok(f"exported {n} SFT pairs -> {path}")
    return path


FINETUNE_SCRIPT = '''#!/usr/bin/env python3
"""
LoRA fine-tune of a small open model on MEGON's exported memory.
This is the honest path from "crawled knowledge base" to "updated weights":
it needs a GPU, and it is NOT run by MEGON itself.

    pip install transformers peft datasets accelerate bitsandbytes
    python3 lora_finetune.py --data megon_home/export/megon-sft-YYYY-MM-DD.jsonl
"""
import argparse, json
import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")   # small enough for 1x 16GB
p.add_argument("--out", default="./mythos-lora")
p.add_argument("--epochs", type=float, default=1.0)
p.add_argument("--bs", type=int, default=2)
p.add_argument("--grad-acc", type=int, default=8)
a = p.parse_args()

rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
tok = AutoTokenizer.from_pretrained(a.base)
tok.pad_token = tok.pad_token or tok.eos_token

def fmt(r):
    return f"<|im_start|>user\\n{r['instruction']}\\n{r.get('input','')}<|im_end|>\\n<|im_start|>assistant\\n{r['output']}<|im_end|>"

ds = Dataset.from_list([{"text": fmt(r)} for r in rows]).map(
    lambda x: tok(x["text"], truncation=True, max_length=1024), batched=False,
    remove_columns=["text"])

model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map="auto")
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                         task_type="CAUSAL_LM"))
model.print_trainable_parameters()

Trainer(model=model, args=TrainingArguments(
    output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=a.bs,
    gradient_accumulation_steps=a.grad_acc, learning_rate=2e-4, bf16=True,
    logging_steps=10, save_strategy="epoch", report_to="none"),
    train_dataset=ds, data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
model.save_pretrained(a.out)
tok.save_pretrained(a.out)
print("adapter saved to", a.out)
'''
