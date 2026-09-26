"""MEGON I AI -- the benchmark MEGON grows by itself"""
from __future__ import annotations
from megon.base import *
from megon.memory import *
from megon.mind import *

# ==============================================================================
# 13. EVALUATION -- the benchmark MEGON grows by itself
# ==============================================================================
QA_PATTERNS = [
    (re.compile(r"^([A-Z][\w\-.]{2,60}(?:\s+[A-Z][\w\-.]{1,40}){0,3})\s+(?:is|was)\s+(an?\s+[^.]{8,140})\.", re.M),
     "What is {0}?", "{1}"),
    (re.compile(r"^([A-Z][\w\-.]{2,50}(?:\s+[A-Z][\w\-.]{1,30}){0,2})\s+was\s+founded\s+in\s+(\d{4})\.", re.M),
     "When was {0} founded?", "{1}"),
    (re.compile(r"^([A-Z][\w\-.]{2,50}(?:\s+[A-Z][\w\-.]{1,30}){0,2})\s+is\s+located\s+in\s+([A-Z][^.]{3,60})\.", re.M),
     "Where is {0} located?", "{1}"),
    (re.compile(r"^([A-Z][\w\-.]{2,50})\s+was\s+born\s+in\s+(\d{4})\.", re.M),
     "In what year was {0} born?", "{1}"),
    (re.compile(r"^In\s+(\d{4}),\s+([A-Z][\w\-.]{2,50})\s+([a-z][^.]{10,120})\.", re.M),
     "What did {1} do in {0}?", "{2}"),
]

MATH_BANK = [
    ("Nadia has 18 marbles. She buys 3 more boxes of 12 marbles each. How many marbles does she have in total?", 54),
    ("A shop sold 45 apples on Monday and 27 fewer apples on Tuesday. How many apples were sold on Tuesday?", 18),
    ("Tom reads 15 pages a day. How many pages does he read in 2 weeks?", 210),
    ("A bus has 48 seats. If 3/8 of the seats are empty, how many seats are occupied?", 30),
    ("Rina earned $240. She spent 25 percent of it on books. How much money is left?", 180),
    ("There are 6 boxes with 14 pencils in each. 9 pencils are broken. How many pencils are usable?", 75),
    ("A tank holds 250 litres. It is filled by a pipe at 25 litres per minute for 6 minutes. How much is left to fill?", 100),
    ("Sam has twice as many cards as Jo. Jo has 37 cards. How many cards do they have together?", 111),
    ("A baker made 120 cakes and sold 45 in the morning and 38 in the afternoon. How many are left?", 37),
    ("Each shelf holds 24 books. There are 7 shelves and 13 books are lent out. How many books remain?", 155),
    ("A car travels 60 km per hour for 3 hours and then 80 km per hour for 2 hours. How many km in total?", 340),
    ("Maya paid $18 for 3 identical notebooks. How much do 7 notebooks cost?", 42),
    ("A factory makes 500 units a day. How many units in 5 working days minus the 120 defective ones?", 2380),
    ("There are 96 students. 1/4 go by bus and 20 walk. How many come by bike?", 52),
    ("A rope is 45 m long. Pieces of 3 m are cut from it, 12 pieces in total. How many metres remain?", 9),
    ("Ivan had 84 coins. He gave away 19 and found 25 more. How many coins does he have now?", 90),
    ("A pack of 12 pens costs $18. What is the cost of 5 packs?", 90),
    ("A farmer collected 340 eggs. 5 percent were cracked. How many eggs are uncracked?", 323),
    ("A train leaves at 9:15 and arrives 4 hours and 40 minutes later. What time is it on arrival, in minutes past midnight?", 835),
    ("Lily saves $35 a week for 8 weeks and then spends $120. How much does she have left?", 160),
]


def token_f1(pred: str, gold: str) -> float:
    p, g = set(words(pred)), set(words(gold))
    if not p or not g:
        return 0.0
    inter = len(p & g)
    if not inter:
        return 0.0
    prec, rec = inter / len(p), inter / len(g)
    return 2 * prec * rec / (prec + rec)


class Evaluator:
    def __init__(self, store: Store, retriever: Retriever, answerer: Answerer, cfg: Dict[str, Any]):
        self.store, self.retriever, self.answerer, self.cfg = store, retriever, answerer, cfg

    # --- self-supervised item generation from freshly ingested text -----------
    def generate_items(self, max_items: int = 40) -> Dict[str, int]:
        c = self.store.c
        made = collections.Counter()
        rows = c.execute(
            """SELECT c.id cid, c.text, c.doc_id, d.split FROM chunks c JOIN docs d ON d.id=c.doc_id
               WHERE d.split='eval' AND c.words BETWEEN 25 AND 90
                 AND c.id NOT IN (SELECT evidence_chunk FROM eval_items WHERE evidence_chunk IS NOT NULL)
               ORDER BY RANDOM() LIMIT ?""", (max_items * 6,)).fetchall()
        vocab_pool = [w for r in c.execute(
            "SELECT text FROM chunks ORDER BY RANDOM() LIMIT 300").fetchall() for w in content_words(r["text"])]
        vocab_pool = [w for w in vocab_pool if len(w) >= 5]
        random.shuffle(vocab_pool)
        for r in rows:
            if sum(made.values()) >= max_items:
                break
            text = r["text"]
            sents = [s for s in sentences(text) if 8 <= len(words(s)) <= 40]
            if not sents:
                continue
            s = random.choice(sents)
            toks = content_words(s)
            cand = [w for w in toks if len(w) >= 6]
            # cloze item
            if cand and made["cloze"] < max_items * 0.45:
                gold = random.choice(sorted(set(cand), key=lambda w: -self.retriever.idf(w))[:4])
                masked = re.sub(r"(?i)\b" + re.escape(gold) + r"\w*\b", "____", s, count=1)
                opts = {gold}
                while len(opts) < 4 and vocab_pool:
                    opts.add(vocab_pool.pop())
                opts = list(opts)[:4]
                random.shuffle(opts)
                c.execute("""INSERT INTO eval_items(kind,question,answer,options,evidence_chunk,
                             source_doc,split,created_at,meta) VALUES(?,?,?,?,?,?,?,?,?)""",
                          ("cloze", masked, gold, json.dumps(opts), r["cid"], r["doc_id"],
                           "eval", now_iso(), json.dumps({"sentence": s})))
                made["cloze"] += 1
            # factoid QA items
            for rx, qtpl, atpl in QA_PATTERNS:
                if made["qa"] >= max_items * 0.35:
                    break
                m = rx.search(s)
                if not m:
                    continue
                g = m.groups()
                q = qtpl.format(*g).strip()
                a = atpl.format(*g).strip()
                if len(words(a)) < 1 or len(q) < 12:
                    continue
                c.execute("""INSERT INTO eval_items(kind,question,answer,options,evidence_chunk,
                             source_doc,split,created_at,meta) VALUES(?,?,?,?,?,?,?,?,?)""",
                          ("qa", q, a, None, r["cid"], r["doc_id"], "eval", now_iso(), "{}"))
                made["qa"] += 1
                break
        # retrieval items from doc titles
        for d in c.execute("SELECT id,title FROM docs WHERE split='eval' AND length(title)>12 "
                           "ORDER BY RANDOM() LIMIT 15").fetchall():
            title = re.sub(r"\s*[-|–].*$", "", d["title"]).strip()
            if len(words(title)) < 2 or title.lower().startswith("http"):
                continue
            cid = c.execute("SELECT id FROM chunks WHERE doc_id=? ORDER BY ord LIMIT 1", (d["id"],)).fetchone()
            c.execute("""INSERT INTO eval_items(kind,question,answer,options,evidence_chunk,
                         source_doc,split,created_at,meta) VALUES(?,?,?,?,?,?,?,?,?)""",
                      ("retrieval", title, str(d["id"]), None, cid["id"] if cid else None, d["id"],
                       "eval", now_iso(), "{}"))
            made["retrieval"] += 1
        # curated arithmetic bank: seeded once, on its own key. It must NOT be
        # gated on kind='math' being empty -- real GSM8K rows arrive during
        # acquisition and would silently suppress the bank forever.
        if c.execute("SELECT COUNT(*) n FROM eval_items WHERE kind='math_bank'").fetchone()["n"] == 0:
            for q, a in MATH_BANK:
                c.execute("""INSERT INTO eval_items(kind,question,answer,options,evidence_chunk,
                             source_doc,split,created_at,meta) VALUES(?,?,?,?,?,?,?,?,?)""",
                          ("math_bank", q, str(a), None, None, None, "eval", now_iso(),
                           '{"seed":true}'))
                made["math_bank"] += 1
        return dict(made)

    # --- run ------------------------------------------------------------------
    def run(self, overrides: Optional[Dict[str, Any]] = None, limit: int = 250,
            verbose: bool = False) -> Dict[str, Any]:
        c = self.store.c
        items = c.execute("SELECT * FROM eval_items ORDER BY id LIMIT ?", (limit,)).fetchall()
        if not items:
            return {"n": 0, "score": 0.0, "note": "no eval items yet"}
        per: Dict[str, List[float]] = collections.defaultdict(list)
        strategy_hits: Dict[str, List[float]] = collections.defaultdict(list)
        weak_topics: List[str] = []
        for it in items:
            kind = it["kind"]
            # Auto-generated items hold out their evidence chunk so the model
            # cannot simply echo it. Real reading-comprehension items (SQuAD)
            # are the opposite: the passage IS the input, so it stays.
            try:
                is_real = bool(json.loads(it["meta"] or "{}").get("real"))
            except Exception:
                is_real = False
            excl = [] if is_real else ([it["evidence_chunk"]] if it["evidence_chunk"] else [])
            score = 0.0
            detail: Dict[str, Any] = {}
            try:
                if kind == "cloze":
                    opts = json.loads(it["options"] or "[]")
                    if not opts:
                        continue
                    scores = []
                    for o in opts:
                        filled = it["question"].replace("____", o)
                        hits = self.retriever.search(filled, k=3, exclude=excl, overrides=overrides)
                        s = sum(h["score"] * (1.0 + 0.5 * h["coverage"]) for h in hits) / max(1, len(hits))
                        scores.append(s)
                    pick = opts[int(np.argmax(scores))] if np is not None else \
                        max(zip(opts, scores), key=lambda t: t[1])[0]
                    score = 1.0 if pick == it["answer"] else 0.0
                    detail = {"picked": pick, "gold": it["answer"]}
                elif kind == "qa":
                    ans = self.answerer.answer(it["question"], exclude=excl, overrides=overrides)
                    f1 = token_f1(ans["answer"], it["answer"])
                    score = f1
                    per["qa_em"].append(1.0 if words(it["answer"]) and
                                        all(w in words(ans["answer"]) for w in words(it["answer"])) else 0.0)
                    detail = {"pred": ans["answer"][:200], "gold": it["answer"], "f1": round(f1, 3)}
                elif kind == "retrieval":
                    hits = self.retriever.search(it["question"], k=3, exclude=[], overrides=overrides)
                    ids = [h["doc_id"] for h in hits]
                    score = 1.0 if ids and ids[0] == int(it["answer"]) else 0.0
                    per["retrieval_r3"].append(1.0 if int(it["answer"]) in ids else 0.0)
                    detail = {"gold_doc": it["answer"], "got": ids}
                elif kind in ("math", "math_bank"):
                    sol = self.answerer.solver.solve(it["question"])
                    gold = float(it["answer"])
                    score = 1.0 if sol["answer"] is not None and abs(sol["answer"] - gold) < 1e-6 else 0.0
                    if sol["strategy"] != "none":
                        strategy_hits[sol["strategy"]].append(score)
                    detail = {"pred": sol["answer"], "gold": gold, "expr": sol["expr"],
                              "strategy": sol["strategy"]}
            except Exception as e:
                detail = {"error": f"{type(e).__name__}: {e}"}
                score = 0.0
            per[kind].append(score)
            if score < 0.34:
                weak_topics.append(" ".join(content_words(it["question"])[:6]))
            c.execute("INSERT INTO eval_results(ts,cycle,item_id,score,detail) VALUES(?,?,?,?,?)",
                      (now_iso(), int(kv_get(c, "cycle", 0)), int(it["id"]), score,
                       json.dumps(detail, ensure_ascii=False, default=str)[:900]))
            if verbose:
                log.d(f"  [{kind}] {score:.2f}  {it['question'][:70]}")

        def avg(name: str) -> float:
            return float(np.mean(per[name])) if per.get(name) else 0.0

        n_math = len(per.get("math", []))
        coverage = avg("retrieval") * 0.5 + avg("retrieval_r3") * 0.5
        comp = (0.24 * avg("cloze") + 0.24 * avg("qa") + 0.18 * avg("retrieval")
                + 0.10 * avg("math") + 0.14 * avg("math_bank") + 0.10 * coverage)
        return {
            "n": len(items), "score": round(float(comp), 4),
            "cloze_acc": round(avg("cloze"), 4), "qa_f1": round(avg("qa"), 4),
            "qa_em": round(avg("qa_em"), 4), "retrieval_r1": round(avg("retrieval"), 4),
            "retrieval_r3": round(avg("retrieval_r3"), 4), "math_acc": round(avg("math"), 4),
            "coverage": round(float(coverage), 4),
            "math_bank_acc": round(avg("math_bank"), 4),
            "by_kind": {k: round(float(np.mean(v)), 4) for k, v in per.items()},
            "strategy_win": {k: round(float(np.mean(v)), 4) for k, v in strategy_hits.items() if v},
            "strategy_n": {k: len(v) for k, v in strategy_hits.items()},
            "weak_topics": collections.Counter(weak_topics).most_common(8),
        }
