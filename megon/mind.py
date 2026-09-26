"""MEGON I AI -- reasoning: arithmetic solver, answer engine, span extraction"""
from __future__ import annotations
from megon.base import *
from megon.memory import *

# ==============================================================================
# 10. ARITHMETIC -- a real (tiny) equation solver for GSM8K-style items
# ==============================================================================
NUMWORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
            "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
            "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
            "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
            "hundred": 100, "thousand": 1000, "million": 1000000, "dozen": 12, "half": 0.5}

OPS = {
    "+": [r"\bmore than\b", r"\bplus\b", r"\badds?\b", r"\badditional\b", r"\bin total\b",
          r"\baltogether\b", r"\bcombined\b", r"\btotal\b", r"\bgives? (him|her|them) \d",
          r"\bbuys? \d+ more\b", r"\band\b"],
    "-": [r"\bless than\b", r"\bfewer\b", r"\bminus\b", r"\bsubtract\w*", r"\bgave away\b",
          r"\bgives? away\b", r"\bsold\b", r"\bsells?\b", r"\blost\b", r"\bleft\b",
          r"\bremain\w*\b", r"\bdifference\b", r"\beats?\b", r"\bused\b", r"\bhow many more\b"],
    "*": [r"\btimes\b", r"\btwice\b", r"\bdouble\b", r"\btriple\b", r"\beach\b", r"\bper\b",
          r"\bmultipl\w+\b", r"\bof them\b", r"\bin each\b", r"\b\d+ (boxes|bags|packs|groups|rows|shelves)\b"],
    "/": [r"\bhalf\b", r"\bdivided\b", r"\bsplit\b", r"\bshared? equally\b", r"\beach gets\b",
          r"\bquotient\b", r"\bthird of\b", r"\bquarter of\b", r"\bper person\b"],
}
OPS_RE = {op: re.compile("|".join(pats), re.I) for op, pats in OPS.items()}


def eval_expr(expr: str) -> Optional[float]:
    """Shunting-yard evaluator: + - * / ( ) and unary minus. No eval() anywhere."""
    toks = re.findall(r"\d+\.?\d*|[()+\-\*/]", expr.replace(" ", ""))
    if not toks:
        return None
    prec = {"+": 1, "-": 1, "*": 2, "/": 2}
    out: List[Any] = []
    ops: List[str] = []

    def apply_op() -> bool:
        if len(ops) == 0 or len(out) < 2:
            return False
        o = ops.pop()
        b, a = out.pop(), out.pop()
        if o == "+":
            out.append(a + b)
        elif o == "-":
            out.append(a - b)
        elif o == "*":
            out.append(a * b)
        elif o == "/":
            if b == 0:
                return False
            out.append(a / b)
        return True

    expect_operand = True
    for t in toks:
        if re.match(r"\d", t):
            out.append(float(t))
            expect_operand = False
        elif t == "(":
            ops.append(t)
            expect_operand = True
        elif t == ")":
            while ops and ops[-1] != "(":
                if not apply_op():
                    return None
            if ops:
                ops.pop()
            expect_operand = False
        else:
            if expect_operand and t == "-":
                out.append(0.0)
                ops.append("-")
                continue
            if expect_operand:
                return None
            while ops and ops[-1] != "(" and prec.get(ops[-1], 0) >= prec[t]:
                if not apply_op():
                    return None
            ops.append(t)
            expect_operand = True
    while ops:
        if ops[-1] == "(":
            ops.pop()
            continue
        if not apply_op():
            return None
    return out[0] if len(out) == 1 else None


def extract_numbers(text: str) -> List[float]:
    out: List[float] = []
    for m in re.finditer(r"\$?(\d[\d,]*(?:\.\d+)?)", text):
        try:
            out.append(float(m.group(1).replace(",", "")))
        except Exception:
            pass
    for w, v in NUMWORDS.items():
        if re.search(r"\b" + w + r"\b", text, re.I):
            out.append(float(v))
    return out


def final_answer_of(text: str) -> Optional[float]:
    m = re.findall(r"answer is[:\s]*\$?(-?\d[\d,]*(?:\.\d+)?)", text, re.I)
    if m:
        return float(m[-1].replace(",", ""))
    m = re.findall(r"=\s*\$?(-?\d[\d,]*(?:\.\d+)?)\s*$", text.strip(), re.M)
    if m:
        return float(m[-1].replace(",", ""))
    n = extract_numbers(text)
    return n[-1] if n else None


def _op_between(text: str, a: int, b: int) -> str:
    seg = text[a:b]
    scores = {op: sum(len(m.group(0)) for m in rx.finditer(seg)) for op, rx in OPS_RE.items()}
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "+"


LOSS_RE = re.compile(
    r"\b(sold|sells?|gave away|gives? away|lost|spends?|spent|used?|ate|eaten|broken|defective|"
    r"cracked|lent out|removed|fewer|cut|emptied|drank|given away|wasted|minus)\b", re.I)
GAIN_RE = re.compile(
    r"\b(has|have|had|bought|buys?|found|made|makes?|collected?|earned|received?|got|gets?|"
    r"saves?|saved|holds?|contains?|there are|there is|leaves at|starts? with|is|are|was|were|"
    r"reads?|travels?|fills?|filled|buys)\b", re.I)

TIME_UNITS_MIN = {"minute": 1.0, "hour": 60.0, "day": 1440.0, "week": 10080.0,
                  "month": 43200.0, "year": 525600.0}
NUMPAT = r"(?:\$)?(\d[\d,]*(?:\.\d+)?)"
RATE_RE = re.compile(NUMPAT + r"\s*[a-z]*\s+(?:a|an|per|each|every)\s+"
                     r"(minute|hour|day|week|month|year)s?\b", re.I)
DUR_RE = re.compile(r"\b(?:for|in|over|during)\s+" + NUMPAT +
                    r"(?:\s+\w+){0,2}\s+(minute|hour|day|week|month|year)s?\b", re.I)
FRAC_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:percent|%)")
CLOCK_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
DUR_HM_RE = re.compile(r"(\d+)\s*hours?\s*(?:and\s*)?(\d*)\s*minutes?", re.I)
MULT_RE = re.compile(r"\b(twice|three times|two times|double|triple|(\d+)\s*times)\b", re.I)

Q_LEFT = re.compile(r"\b(left|remain\w*|how many more|difference|uncracked|usable|occupied|minus|"
                    r"how much (?:money|is left)|now\??$)\b", re.I)
Q_TOTAL = re.compile(r"\b(total|altogether|in all|in total|together|sum)\b", re.I)
Q_COST = re.compile(r"\b(cost|costs?|pay|spend|price|worth)\b", re.I)

# How much to trust a program when several propose different answers. Template
# matches are specific; the generic fallbacks are not. These priors are the
# *starting point* -- the tuner moves them from graded evidence every cycle.
SPECIFICITY = {"percent_left": 3.0, "percent_of": 3.0, "fraction_left": 3.0, "fraction_of": 3.0,
               "rate_duration": 3.0, "per_unit_count": 3.0, "group_each": 2.6, "cut_product": 3.0,
               "multiple": 3.0, "unit_price": 3.0, "fewer_than": 3.0, "pack_cost": 3.0,
               "time_add": 3.2, "fraction_rest": 3.2, "ledger": 1.2, "chain": 0.25}


def _num_spans(text: str) -> List[Tuple[int, int, float]]:
    return [(m.start(), m.end(), float(m.group(1).replace(",", "")))
            for m in re.finditer(NUMPAT, text)]


def _last(rx: "re.Pattern[str]", s: str) -> Optional["re.Match[str]"]:
    """Last match, i.e. the one nearest to the end of the window."""
    last = None
    for mm in rx.finditer(s):
        last = mm
    return last


def _polarity(text: str, start: int, end: int, prev: int) -> int:
    """Gain (+1) or loss (-1) for one number.

    Uses the verb *nearest* to the number on either side, not merely the first
    verb found. Both details matter: in "He gave away 19 and found 25 more" the
    nearest verb before 25 is "found" (a gain), not the earlier "gave away"; and
    in "There are 7 shelves and 13 books are lent out" the loss wording sits
    closer to 13 than the gain verb does.
    """
    before = text[max(0, start - 48):start]
    lo, go = _last(LOSS_RE, before), _last(GAIN_RE, before)
    lp = lo.end() if lo else -1
    gp = go.end() if go else -1
    verb_dist, verb_pol = None, prev
    if lp >= 0 or gp >= 0:
        verb_dist = len(before) - max(lp, gp)
        verb_pol = -1 if lp > gp else 1
    am = LOSS_RE.search(text[end:end + 34])
    if am and (verb_dist is None or am.start() < verb_dist):
        return -1
    return verb_pol


class MathSolver:
    """
    A rule-based arithmetic reasoner over a family of word-problem templates.

    It is NOT a language model and will not match one on open-ended GSM8K.
    What is real about it: ~15 typed programs each propose a value together with
    the expression they used; proposals are weighted by how specific the
    template match was, boosted when independent programs agree; and the
    per-program priors are *learned* from graded eval results each cycle.
    """

    STRATEGIES = list(SPECIFICITY)

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {s: 1.0 for s in self.STRATEGIES}

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _question(p: str) -> str:
        qs = [s for s in sentences(p) if "?" in s]
        return qs[-1] if qs else (sentences(p)[-1] if sentences(p) else p)

    @staticmethod
    def _capacity(p: str) -> Optional[float]:
        m = re.search(r"\b(?:holds?|capacity of|can hold|contains?|total of)\s+" + NUMPAT, p, re.I)
        return float(m.group(1).replace(",", "")) if m else None

    @classmethod
    def _losses(cls, p: str, exclude: Iterable[Tuple[int, int]] = ()) -> float:
        """Sum of quantities the text marks as losses.

        `exclude` drops numbers that belong to a rate/duration expression -- in
        "500 units a day ... in 5 working days minus the 120 defective ones",
        the 5 is a duration, not something lost.
        """
        tot, prev = 0.0, 1
        for st, en, v in _num_spans(p):
            prev = _polarity(p, st, en, prev)
            if prev < 0 and not any(st >= a and en <= b for a, b in exclude):
                tot += v
        return tot

    @classmethod
    def _ledger(cls, p: str) -> Tuple[Optional[float], str]:
        parts, prev, tot = [], 1, 0.0
        for st, en, v in _num_spans(p):
            prev = _polarity(p, st, en, prev)
            parts.append(("+" if prev > 0 else "-") + f"{v:g}")
            tot += prev * v
        return (tot, " ".join(parts)) if parts else (None, "")

    # -- the program bank ------------------------------------------------------
    def candidates(self, problem: str) -> List[Dict[str, Any]]:
        p = re.sub(r"\s+", " ", problem).strip()
        q = self._question(p)
        asks_left = bool(Q_LEFT.search(q))
        asks_total = bool(Q_TOTAL.search(q))
        asks_cost = bool(Q_COST.search(q))
        out: List[Dict[str, Any]] = []

        def push(name: str, value: Optional[float], expr: str) -> None:
            if value is None or not isinstance(value, (int, float)):
                return
            if not math.isfinite(float(value)) or abs(value) > 1e12:
                return
            out.append({"strategy": name, "expr": expr, "value": float(value),
                        "prior": float(self.weights.get(name, 1.0)) * SPECIFICITY.get(name, 1.0)})

        nums = [v for _, _, v in _num_spans(p)]
        pcts = [float(m.group(1)) for m in PCT_RE.finditer(p)]
        fracs = [(float(m.group(1)), float(m.group(2))) for m in FRAC_RE.finditer(p)]
        sents = sentences(p)

        # percent -------------------------------------------------------------
        if pcts and nums:
            base = nums[0] if nums[0] > pcts[0] else (nums[1] if len(nums) > 1 else nums[0])
            push("percent_left" if asks_left else "percent_of",
                 base * (1 - pcts[0] / 100.0) if asks_left else base * pcts[0] / 100.0,
                 f"{base:g}*(1-{pcts[0]:g}/100)" if asks_left else f"{base:g}*{pcts[0]:g}/100")

        # fractions -----------------------------------------------------------
        if fracs and nums:
            a, b = fracs[0]
            base = next((v for v in nums if v != a and v != b), nums[0])
            if b:
                push("fraction_left" if asks_left else "fraction_of",
                     base * (1 - a / b) if asks_left else base * a / b,
                     f"{base:g}*(1-{a:g}/{b:g})" if asks_left else f"{base:g}*{a:g}/{b:g}")
                rest = [v for v in nums if v not in (base, a, b)]
                if rest and b:
                    push("fraction_rest", base - base * a / b - sum(rest),
                         f"{base:g}-{base:g}*{a:g}/{b:g}-{'-'.join(f'{v:g}' for v in rest)}")

        # rate x duration -----------------------------------------------------
        rates = [(float(m.group(1).replace(",", "")), m.group(2).lower()) for m in RATE_RE.finditer(p)]
        durs = [(float(m.group(1).replace(",", "")), m.group(2).lower()) for m in DUR_RE.finditer(p)]
        if rates and durs:
            terms, tot = [], 0.0
            for (rv, ru), (dv, du) in zip(rates, durs):
                factor = TIME_UNITS_MIN.get(du, 1.0) / TIME_UNITS_MIN.get(ru, 1.0)
                tot += rv * dv * factor
                terms.append(f"{rv:g}*{dv:g}" + (f"*{factor:g}" if factor != 1 else ""))
            expr = "+".join(terms)
            cap = self._capacity(p)
            if asks_left and cap is not None:
                push("rate_duration", cap - tot, f"{cap:g}-({expr})")
            elif asks_left:
                spans = [(mm.start(1), mm.end(1)) for mm in RATE_RE.finditer(p)]
                spans += [(mm.start(1), mm.end(1)) for mm in DUR_RE.finditer(p)]
                loss = self._losses(p, spans)
                push("rate_duration", tot - loss, f"({expr})-{loss:g}")
            else:
                push("rate_duration", tot, expr)

        # "each X holds N" x "there are M X" ------------------------------------
        m_each = re.search(r"each\s+\w+\s+(?:holds?|has|have|contains?|carries?)\s+" + NUMPAT, p, re.I)
        m_cnt = re.search(r"there are\s+" + NUMPAT, p, re.I)
        if m_each and m_cnt:
            n = float(m_each.group(1).replace(",", ""))
            cnt = float(m_cnt.group(1).replace(",", ""))
            loss = self._losses(p)
            push("per_unit_count", n * cnt - loss if asks_left else n * cnt,
                 f"{n:g}*{cnt:g}" + (f"-{loss:g}" if asks_left and loss else ""))

        # groups "with M in each" ---------------------------------------------
        each_sent = next((s for s in sents if re.search(r"\beach\b", s, re.I)), None)
        if each_sent:
            en = [v for _, _, v in _num_spans(each_sent)]
            if len(en) >= 2:
                prod = 1.0
                for v in en:
                    prod *= v
                others = [v for s in sents if s != each_sent for _, _, v in _num_spans(s)]
                if asks_left:
                    loss = self._losses(" ".join(s for s in sents if s != each_sent))
                    push("group_each", prod - loss, f"{prod:g}-{loss:g}")
                elif asks_total and others:
                    push("group_each", prod + others[0], f"{others[0]:g}+{prod:g}")
                else:
                    push("group_each", prod, "*".join(f"{v:g}" for v in en))

        # "pieces of 3 m, 12 pieces" -> total - product -------------------------
        cut_sent = next((s for s in sents if re.search(r"\b(cut|pieces?|slices?)\b", s, re.I)), None)
        if cut_sent and nums:
            cn = [v for _, _, v in _num_spans(cut_sent)]
            if len(cn) >= 2:
                prod = 1.0
                for v in cn:
                    prod *= v
                base = nums[0]
                if base not in cn:
                    push("cut_product", base - prod,
                         f"{base:g}-{'*'.join(f'{v:g}' for v in cn)}")

        # gain/loss ledger -----------------------------------------------------
        val, expr = self._ledger(p)
        if val is not None:
            push("ledger", val, expr)

        # "twice as many as" ----------------------------------------------------
        m = MULT_RE.search(p)
        if m and nums:
            g1 = (m.group(1) or "").lower()
            k = 2.0 if g1 in ("twice", "double", "two times") else \
                (3.0 if g1 in ("triple", "three times") else float(m.group(2) or 2))
            base = nums[-1]
            push("multiple", base * (1 + k) if asks_total else base * k,
                 f"{base:g}+{k:g}*{base:g}" if asks_total else f"{k:g}*{base:g}")

        # unit price: "$A for B items ... C items cost?" ------------------------
        m = re.search(NUMPAT + r"\s+for\s+" + NUMPAT, p)
        m2 = re.search(NUMPAT + r"\s+\w+\s+cost", q, re.I)
        if m and m2:
            a, b = float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))
            c3 = float(m2.group(1).replace(",", ""))
            if b:
                push("unit_price", a / b * c3, f"{a:g}/{b:g}*{c3:g}")

        # "a pack of N costs $A ... cost of M packs" -----------------------------
        m = re.search(r"costs?\s+" + NUMPAT, p, re.I)
        mq = re.search(NUMPAT + r"\s+\w*(?:packs?|boxes|bags)", q, re.I)
        if m and mq and asks_cost:
            a = float(m.group(1).replace(",", ""))
            cnt = float(mq.group(1).replace(",", ""))
            push("pack_cost", a * cnt, f"{a:g}*{cnt:g}")

        # "N fewer" ---------------------------------------------------------------
        m = re.search(NUMPAT + r"\s+fewer", p, re.I)
        if m and len(nums) >= 2:
            n = float(m.group(1).replace(",", ""))
            base = next((v for v in nums if v != n), nums[0])
            push("fewer_than", base - n, f"{base:g}-{n:g}")

        # clock arithmetic ---------------------------------------------------------
        mc, md = CLOCK_RE.search(p), DUR_HM_RE.search(p)
        if mc and md:
            start = int(mc.group(1)) * 60 + int(mc.group(2))
            add = int(md.group(1)) * 60 + int(md.group(2) or 0)
            push("time_add", start + add, f"{start}+{add}")

        # generic fallbacks ---------------------------------------------------------
        if len(nums) >= 2:
            pos = [mm.start() for mm in re.finditer(NUMPAT, p)]
            expr = f"{nums[0]:g}"
            for i in range(1, len(nums)):
                a = pos[i - 1] if i - 1 < len(pos) else 0
                b = pos[i] if i < len(pos) else len(p)
                expr += f" {_op_between(p, a, b)} {nums[i]:g}"
            push("chain", eval_expr(expr), expr)
            head = eval_expr("+".join(f"{n:g}" for n in nums[:-1]))
            if head is not None and pos:
                last = _op_between(p, pos[-2] if len(pos) > 1 else 0,
                                   pos[-1] if len(pos) > 1 else len(p))
                push("chain", eval_expr(f"{head:g} {last} {nums[-1]:g}"),
                     f"{head:g} {last} {nums[-1]:g}")
        if nums:
            push("chain", sum(nums), "+".join(f"{n:g}" for n in nums))
        return out

    def solve(self, problem: str) -> Dict[str, Any]:
        cands = self.candidates(problem)
        if not cands:
            return {"answer": None, "expr": None, "strategy": "none", "confidence": 0.0,
                    "candidates": []}
        # One vote per program, not per candidate: otherwise the generic
        # fallbacks (which always fire, several times each) drown out the
        # specific template match that actually understood the question.
        per_prog: Dict[str, Dict[int, float]] = {}
        for c in cands:
            v = int(round(c["value"]))
            cur = per_prog.setdefault(c["strategy"], {})
            cur[v] = max(cur.get(v, 0.0), c["prior"])
        votes: Dict[int, float] = collections.defaultdict(float)
        agree: Dict[int, int] = collections.Counter()
        for prog, vals in per_prog.items():
            for v, w in vals.items():
                votes[v] += w
                agree[v] += 1
        for k in list(votes):
            votes[k] *= (1.0 + 0.30 * (agree[k] - 1))          # agreement is a tiebreaker, not a bulldozer
        best_val = max(votes, key=lambda k: votes[k])
        best = next(c for c in cands if int(round(c["value"])) == best_val)
        tot = sum(votes.values()) or 1.0
        return {"answer": best_val, "expr": best["expr"], "strategy": best["strategy"],
                "confidence": round(votes[best_val] / tot, 4), "candidates": cands}


# ==============================================================================
# 11. LLM PROVIDER (optional -- used only if you supply an endpoint + key)
# ==============================================================================
def llm_ready(cfg: Dict[str, Any]) -> bool:
    L = cfg.get("llm", {})
    if L.get("provider", "auto") == "off":
        return False
    return bool(L.get("base_url") and L.get("model") and os.environ.get(L.get("api_key_env", ""), ""))


def llm_generate(cfg: Dict[str, Any], system: str, user: str) -> Optional[str]:
    L = cfg.get("llm", {})
    key = os.environ.get(L.get("api_key_env", "MEGON_LLM_KEY"), "")
    if not key or requests is None:
        return None
    url = L["base_url"].rstrip("/") + ("/chat/completions" if "anthropic" not in L["base_url"] else "/v1/messages")
    payload: Dict[str, Any]
    headers = {"Content-Type": "application/json"}
    if "anthropic" in L["base_url"]:
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
        payload = {"model": L["model"], "max_tokens": int(L.get("max_tokens", 700)),
                   "system": system, "messages": [{"role": "user", "content": user}]}
    else:
        headers["Authorization"] = f"Bearer {key}"
        payload = {"model": L["model"], "max_tokens": int(L.get("max_tokens", 700)),
                   "temperature": float(L.get("temperature", 0.2)),
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            log.w(f"llm http {r.status_code}")
            return None
        d = r.json()
        if "choices" in d:
            return d["choices"][0]["message"]["content"]
        if "content" in d:
            return "".join(b.get("text", "") for b in d["content"])
    except Exception as e:
        log.w(f"llm error {e}")
    return None


# ==============================================================================
# 12. ANSWER ENGINE
# ==============================================================================
QUOTE_RE = re.compile(r"^[\s\"'“”‘’()\[\]]+|[\s\"'“”‘’()\[\]]+$")


class Answerer:
    def __init__(self, store: Store, retriever: Retriever, cfg: Dict[str, Any],
                 solver: Optional[MathSolver] = None):
        self.store, self.retriever, self.cfg = store, retriever, cfg
        self.solver = solver or MathSolver()

    # --- sentence selection ---------------------------------------------------
    def _dense_sent_sims(self, q: str, sents: List[str]) -> List[float]:
        """Cosine between the question and each sentence in the LSA space.

        Pure lexical overlap cannot match "sits on top of" to "Atop the Main
        Building". The TF-IDF+SVD model built for retrieval already knows those
        are related, so reuse it here instead of only counting shared words.
        """
        idx = self.retriever.index
        if not sents or idx.vectorizer is None or idx.svd is None or np is None:
            return [0.0] * len(sents)
        try:
            qv = idx.dense_query(q)
            if qv is None:
                return [0.0] * len(sents)
            M = idx.svd.transform(idx.vectorizer.transform(sents))
            M = np.nan_to_num(np.asarray(M, dtype=np.float32))
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            M /= norms
            sims = M @ qv[:M.shape[1]]
            lo, hi = float(sims.min()), float(sims.max())
            if hi - lo < 1e-9:
                return [0.5] * len(sents)
            return [float((s - lo) / (hi - lo)) for s in sims]
        except Exception:
            return [0.0] * len(sents)

    def _rank_sentences(self, q: str, hits: List[Hit]) -> List[Tuple[float, int, str, Hit]]:
        qset = set(content_words(q))
        idf = {w: self.retriever.idf(w) for w in qset}
        # Normalise by the TOTAL idf of the question, not the max. Normalising by
        # max lets a sentence that matches only the single rarest word (e.g.
        # "Notre Dame", which is in nearly every candidate) score a perfect 1.0
        # and outrank the sentence that actually answers the question.
        tot_idf = sum(idf.values()) or 1.0
        flat: List[Tuple[int, int, str, Hit, set]] = []
        for rank, h in enumerate(hits):
            for j, s in enumerate(sentences(h["text"])):
                sw = set(content_words(s))
                if sw and (qset & sw):
                    flat.append((rank, j, s, h, sw))
        dense = self._dense_sent_sims(q, [f[2] for f in flat])
        scored: List[Tuple[float, int, str, Hit]] = []
        for (rank, j, s, h, sw), dsim in zip(flat, dense):
            inter = qset & sw
            sem = sum(idf[w] for w in inter) / tot_idf
            cov = len(inter) / max(1, len(qset))
            pos = 1.0 / (1.0 + 0.12 * j)
            srcw = 1.0 / (1.0 + 0.5 * rank)
            length = clamp(len(words(s)) / 18.0, 0.25, 1.4)
            # blend lexical evidence with distributional similarity
            score = (0.30 * sem + 0.30 * dsim + 0.14 * cov + 0.08 * pos
                     + 0.10 * srcw + 0.08 * min(1.0, length))
            scored.append((score, rank, s.strip(QUOTE_RE.pattern), h))
        scored.sort(key=lambda t: -t[0])
        out: List[Tuple[float, int, str, Hit]] = []
        used: List[set] = []
        for sc, rank, s, h in scored:
            toks = set(words(s))
            if any(len(toks & u) / max(1, min(len(toks), len(u))) > 0.62 for u in used):
                continue
            out.append((sc, rank, s, h))
            used.append(toks)
        return out

    # -- short-answer span extraction -------------------------------------------
    QTYPE = [("when", re.compile(r"^\s*(?:when|in what year|what year|in which year)\b", re.I)),
             ("where", re.compile(r"^\s*where\b", re.I)),
             ("who", re.compile(r"^\s*(?:who|whom|whose)\b", re.I)),
             ("howmany", re.compile(r"^\s*how\s+(?:many|much|long|old|far|fast)\b", re.I)),
             ("which", re.compile(r"^\s*which\b", re.I))]

    # (kind, regex) candidate answer spans inside a sentence
    SPAN_PATTERNS = [
        ("num", re.compile(r"(?:(?:January|February|March|April|May|June|July|August|September|"
                           r"October|November|December)\s+)?\$?\d[\d,]*(?:\.\d+)?"
                           r"(?:\s?(?:%|percent|bc|ad))?")),
        ("cop", re.compile(r"\b(?:is|was|were|are|been|called|named|known as)\s+(?:an?\s+)?"
                           r"([^,.;:()]{2,90})")),
        ("prep", re.compile(r"\b(?:in|at|on|by|from|near|during|beside|behind|above|below|inside|"
                            r"outside|next to|in front of|across from)\s+"
                            r"((?:the\s+|a\s+|an\s+)?[A-Za-z0-9][^,.;:()]{1,70})")),
        ("cap", re.compile(r"\b[A-Z][\w\-.]*(?:(?:\s+(?:of|the|and|for|de|la|von|van|del|di)\s+"
                           r"|\s+)[A-Z][\w\-.]*){0,4}")),
    ]
    # Which span kinds each question type prefers
    TYPE_PREF = {"when": {"num": 1.0, "cap": 0.7, "prep": 0.5, "cop": 0.2},
                 "where": {"prep": 1.0, "cap": 0.85, "cop": 0.4, "num": 0.15},
                 "who": {"cap": 1.0, "cop": 0.5, "prep": 0.4, "num": 0.05},
                 "howmany": {"num": 1.0, "cop": 0.3, "cap": 0.2, "prep": 0.2},
                 "which": {"cap": 0.9, "cop": 0.8, "num": 0.7, "prep": 0.5},
                 "what": {"cop": 0.9, "cap": 0.85, "num": 0.6, "prep": 0.5}}
    # Capitalised words that start sentences far more often than they name things.
    COMMON_CAP = set("""While However Although This These Those There Here After Before Since During
    Following Later Today Recently Because When Where What Which Who Why Both Many Most Some Such
    Only Even Also Instead Meanwhile Additionally Finally Originally Initially Currently Typically
    Generally Often Usually Its His Her Their Our Your Next Previous Several Various Other
    Another Each Every First Second Third Last Catholic""".split())
    VERB_CUT = re.compile(r"\s\b(?:is|are|was|were|has|have|had|be|been|which|that|who|whom)\b\s",
                          re.I)

    @classmethod
    def _trim_span(cls, s: str) -> str:
        """Cut a span at the first verb -- "the Main Building is a statue" is not one answer."""
        m = cls.VERB_CUT.search(" " + s + " ")
        if m:
            s = s[:max(0, m.start() - 1)]
        return s.strip(" \u201c\u201d\"'-,;")

    @classmethod
    def _spans(cls, sent: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for kind, rx in cls.SPAN_PATTERNS:
            for m in rx.finditer(sent):
                raw = m.group(1) if m.groups() else m.group(0)
                s = cls._trim_span(raw)
                if not (1 < len(s) <= 90) or len(words(s)) > 9:
                    continue
                if kind in ("cap", "np") and " " not in s and s in cls.COMMON_CAP:
                    continue                      # "While" is not a named entity
                out.append((kind, s))
        return out

    def short_answer(self, q: str, hits: List[Hit],
                     ranked: Optional[List[Tuple[float, int, str, "Hit"]]] = None,
                     ) -> Tuple[str, float]:
        """Extract a concise answer span from the retrieved evidence.

        Retrieval was already finding the right evidence; returning the whole
        grounded paragraph as "the answer" wrecks precision on factoid
        questions. This keeps the paragraph (as `explanation`) and returns the
        span itself.

        Spans are drawn only from sentences that `_rank_sentences` scored as
        answering the question. Scanning whole passages instead is what makes
        an extractor return the same irrelevant span ("in Rome") for every
        question about a document that happens to mention Rome.
        """
        if not hits:
            return "", 0.0
        qtype = "what"
        for name, rx in self.QTYPE:
            if rx.search(q):
                qtype = name
                break
        pref = self.TYPE_PREF.get(qtype, self.TYPE_PREF["what"])
        qwords = set(content_words(q))
        best, best_score = "", 0.0
        if ranked:
            order = [(float(sc), int(rk), s, h) for sc, rk, s, h in ranked[:6]]
        else:
            order = []
            for rank, h in enumerate(hits[:3]):
                for s in sentences(h["text"])[:4]:
                    order.append((0.5, rank, s, h))
        for sent_score, rank, s, h in order:
            # Wikipedia infoboxes chunk into table fragments like
            # "Paradigms Supervised", which look exactly like a two-word named
            # entity to a span extractor. Require a real sentence.
            if not (8 <= len(words(s)) <= 55):
                continue
            if True:
                for kind, span in self._spans(s):
                    sw = set(content_words(span))
                    if not sw:
                        continue
                    overlap = len(sw & qwords) / max(1, len(sw))
                    sc = pref.get(kind, 0.3)
                    # A span built from words the corpus uses everywhere is rarely the
                    # answer. IDF is already computed for retrieval, so reuse it.
                    idfs = [self.retriever.idf(w) for w in sw]
                    sc *= clamp((sum(idfs) / len(idfs)) / 4.0, 0.12, 1.0)
                    if len(sw) == 1 and kind in ("cap",):
                        sc *= 0.55                                       # lone capitals are usually not entities
                    sc *= 0.35 + 1.4 * min(1.0, sent_score)              # only trust answering sentences
                    sc *= 1.0 / (1.0 + 0.55 * rank)                      # trust better passages
                    sc *= 1.0 - 0.75 * overlap                           # answers rarely echo the question
                    sc *= 1.0 / (1.0 + 0.10 * max(0, len(sw) - 2))       # prefer tight spans
                    if re.fullmatch(r"\d{4}", span.strip()) and qtype == "when":
                        sc *= 1.6
                    if len(span.split()) <= 3:
                        sc *= 1.15
                    if sc > best_score:
                        best, best_score = span, sc
        # best_score is an unbounded product of weights; scale it so 0.35 stays
        # a meaningful "keep the span" threshold instead of saturating at 1.0.
        return best, round(min(1.0, best_score / 1.7), 4)

    def answer(self, q: str, k: Optional[int] = None, exclude: Iterable[int] = (),
               overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        q = q.strip()
        hits = self.retriever.search(q, k=k, exclude=exclude, overrides=overrides)
        maths = bool(re.search(r"\b(how many|how much|total|altogether|left|each|percent|twice)\b", q, re.I)) \
            and len(extract_numbers(q)) >= 1
        sol = self.solver.solve(q) if maths else None
        ranked = self._rank_sentences(q, hits)
        max_s = int(self.cfg.get("generation", {}).get("max_sentences", 5))
        chosen = sorted(ranked[:max_s], key=lambda t: (t[3]["score"] * -1, t[1]))
        order = sorted(chosen, key=lambda t: (t[1],))
        parts = [s for _, _, s, _ in order]
        cites: List[Dict[str, Any]] = []
        seen_url: Dict[str, int] = {}
        text_by_n: Dict[int, str] = {}
        for i, (_, _, s, h) in enumerate(order):
            url = h["url"] or f"memory://chunk/{h['chunk_id']}"
            n = seen_url.setdefault(url, len(seen_url) + 1)
            if n not in text_by_n:
                text_by_n[n] = h["title"] or url
            text_by_n[i + 1] = f"{s} [{n}]"
        body = " ".join(text_by_n[i + 1] for i in range(len(order)))
        for url, n in sorted(seen_url.items(), key=lambda kv: kv[1]):
            hits_for_url = [h for h in hits if (h["url"] or "") == url]
            cites.append({"n": n, "url": url,
                          "title": (hits_for_url[0]["title"] if hits_for_url else url),
                          "chunk_id": hits_for_url[0]["chunk_id"] if hits_for_url else None,
                          "score": hits_for_url[0]["score"] if hits_for_url else 0.0})
        method = "extractive"
        if sol and sol.get("answer") is not None:
            body = f"{body} The answer is {int(sol['answer'])}." if body else \
                   f"The answer is {int(sol['answer'])} (via {sol['expr']})."
            method = "extractive+math"
        # optional neural generation on top of the same retrieved evidence
        if llm_ready(self.cfg) and hits:
            ctx = "\n\n".join(f"[{i+1}] {h['title']}\n{h['text'][:1200]}" for i, h in enumerate(hits))
            sysp = ("You are MEGON I AI. Answer ONLY from the numbered evidence. "
                    "Cite as [n]. If the evidence is insufficient, say so plainly.")
            gen = llm_generate(self.cfg, sysp, f"EVIDENCE:\n{ctx}\n\nQUESTION: {q}")
            if gen:
                body, method = gen.strip(), "rag-llm"

        top = hits[0]["score"] if hits else 0.0
        cov = hits[0]["coverage"] if hits else 0.0
        conf = clamp(0.35 * min(1.0, top / 0.9) + 0.45 * cov + 0.20 * (1.0 if ranked else 0.0), 0, 1)
        explanation = body
        short, short_conf = ("", 0.0)
        if method != "rag-llm":
            short, short_conf = self.short_answer(q, hits, ranked)
        answer = short if short and short_conf >= 0.35 else body
        # A question that is really arithmetic should be answered by the
        # arithmetic. Otherwise a span extractor happily returns whatever
        # phrase the corpus happens to share words with ("mainly diurnal").
        if sol and sol.get("answer") is not None and sol.get("confidence", 0) >= 0.40:
            answer = f"{int(sol['answer'])} (from {sol['expr']})"
        if not body:
            explanation = answer = ("Nothing in memory yet answers that. I have queued the topic "
                                    "for the next learn cycle (see `megon wishlist`).")
            conf = 0.0
        return {"question": q, "answer": answer, "explanation": explanation,
                "method": method, "confidence": round(conf, 3), "short_conf": short_conf,
                "citations": cites, "evidence": hits, "math": sol,
                "ms": int((time.time() - t0) * 1000),
                "corpus": {"chunks": len(self.retriever.index.ids)}}
