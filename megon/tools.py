"""MEGON I AI -- the tool layer.

This is what gives MEGON hands. Every capability is a plain function with a
declared cost and risk, registered so the agent can *choose* between them, and
gated by the policy envelope before it runs.

All of these work with no API keys:
    DuckDuckGo HTML   general web search
    Wikipedia API     search + full page extracts
    arXiv API         preprints
    OpenAlex          250M scholarly works
    Crossref          DOI / citation metadata
    StackExchange     programming Q&A
    GitHub API        code and repositories
    HF datasets-server  the corpora top models train on

`run_python` is containment, not a security boundary: separate process, hard
timeout, capped output, confined working directory, fully logged. Treat it as
"bounded autonomy", which is the honest description of what it is.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from megon.base import *
from megon.memory import Store, SemanticIndex, chunk_text


# ==============================================================================
# search backends (key-free)
# ==============================================================================
DDG = "https://html.duckduckgo.com/html/"
MOJEEK = "https://www.mojeek.com/search"
WIKI_API = "https://en.wikipedia.org/w/api.php"
HF_ROWS = "https://datasets-server.huggingface.co/rows"
ARXIV = "https://export.arxiv.org/api/query"
OPENALEX = "https://api.openalex.org/works"
CROSSREF = "https://api.crossref.org/works"
STACKEX = "https://api.stackexchange.com/2.3/search/advanced"
GITHUB = "https://api.github.com/search/repositories"

DDG_LINK = re.compile(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S | re.I)
DDG_SNIP = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    return WS.sub(" ", htmllib.unescape(TAG_RE.sub(" ", s))).strip()


def _ddg_unwrap(href: str) -> str:
    """DuckDuckGo wraps outbound links in a redirect; pull the real URL out."""
    if href.startswith("//"):
        href = "https:" + href
    q = urllib.parse.urlparse(href).query
    if "uddg=" in q:
        return urllib.parse.unquote(urllib.parse.parse_qs(q).get("uddg", [""])[0])
    return href


def web_search(query: str, n: int = 8, ctx: "ToolCtx" = None) -> Dict[str, Any]:
    """General web search via DuckDuckGo's HTML endpoint. No key, no quota."""
    data, status = ctx.get(DDG + "?" + urllib.parse.urlencode({"q": query, "kl": "us-en"}),
                           max_bytes=400_000) if ctx else (None, "no-context")
    if data is None:
        return {"results": [], "status": status}
    s = data.decode("utf-8", "replace")
    links = DDG_LINK.findall(s)
    snips = DDG_SNIP.findall(s)
    out = []
    for i, (href, title) in enumerate(links[:n]):
        out.append({"title": _clean(title)[:200], "url": _ddg_unwrap(href),
                    "snippet": _clean(snips[i])[:500] if i < len(snips) else ""})
    if not out:
        # A 200 with no result links is a soft block or a rate limit, not an
        # empty web. Reporting "ok" here would teach the kernel that the tool
        # is useless, when the truth is that the source refused us.
        if len(s) < 40_000 or "anomaly" in s.lower() or "challenge" in s.lower():
            return {"results": [], "count": 0, "status": "blocked:search-rate-limited",
                    "note": "search engine returned no links (soft block); try later"}
        return {"results": [], "count": 0, "status": "no-results",
                "note": "search engine reachable but returned nothing"}
    return {"results": out, "count": len(out), "status": "ok"}


def wikipedia(topic: str, ctx: "ToolCtx" = None) -> Dict[str, Any]:
    """Search Wikipedia and return full plain-text extracts of the top hits."""
    q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": topic,
                                "srlimit": 3, "format": "json"})
    data, status = ctx.get(WIKI_API + "?" + q, max_bytes=200_000) if ctx else (None, "no-ctx")
    if data is None:
        return {"pages": [], "status": status}
    try:
        hits = json.loads(data.decode("utf-8", "replace"))["query"]["search"]
    except Exception as e:
        return {"pages": [], "status": f"parse:{e}"}
    pages = []
    for h in hits:
        qq = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1,
                                     "format": "json", "titles": h["title"], "redirects": 1})
        d2, st = ctx.get(WIKI_API + "?" + qq, max_bytes=1_500_000)
        if d2 is None:
            continue
        try:
            page = list(json.loads(d2.decode("utf-8", "replace"))["query"]["pages"].values())[0]
            txt = page.get("extract") or ""
        except Exception:
            continue
        if len(txt) > 150:
            pages.append({"title": h["title"],
                          "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(h['title'].replace(' ', '_'))}",
                          "text": txt[:60_000], "chars": len(txt)})
    return {"pages": pages, "count": len(pages), "status": "ok"}


def arxiv(query: str, n: int = 5, ctx: "ToolCtx" = None) -> Dict[str, Any]:
    """Preprints. Returns title, authors, abstract, pdf link."""
    q = urllib.parse.urlencode({"search_query": f"all:{query}", "max_results": n,
                                "sortBy": "relevance"})
    data, status = ctx.get(ARXIV + "?" + q, max_bytes=900_000) if ctx else (None, "no-ctx")
    if data is None:
        return {"papers": [], "status": status}
    s = data.decode("utf-8", "replace")
    out = []
    for entry in re.findall(r"<entry>(.*?)</entry>", s, re.S):
        def g(tag: str) -> str:
            m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", entry, re.S)
            return _clean(m.group(1)) if m else ""
        out.append({"title": g("title"), "abstract": g("summary")[:2500],
                    "authors": [a for a in re.findall(r"<name>(.*?)</name>", entry)][:8],
                    "url": g("id"), "published": g("published")[:10]})
    return {"papers": out, "count": len(out), "status": "ok"}


def papers(query: str, n: int = 5, ctx: "ToolCtx" = None) -> Dict[str, Any]:
    """OpenAlex: 250M scholarly works, citations, venues. Key-free."""
    q = urllib.parse.urlencode({"search": query, "per-page": n,
                                "select": "id,doi,title,publication_year,cited_by_count,"
                                          "abstract_inverted_index,primary_location"})
    data, status = ctx.get(OPENALEX + "?" + q) if ctx else (None, "no-ctx")
    if data is None:
        return {"works": [], "status": status}
    try:
        rows = json.loads(data.decode("utf-8", "replace")).get("results", [])
    except Exception as e:
        return {"works": [], "status": f"parse:{e}"}
    out = []
    for r in rows:
        out.append({"title": r.get("title"), "year": r.get("publication_year"),
                    "citations": r.get("cited_by_count", 0), "doi": r.get("doi"),
                    "abstract": _uninvert(r.get("abstract_inverted_index"))[:1800]})
    return {"works": out, "count": len(out), "status": "ok"}


def _uninvert(inv: Optional[Dict[str, List[int]]]) -> str:
    """OpenAlex ships abstracts as an inverted index. Rebuild the text."""
    if not inv:
        return ""
    pos: Dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def code_search(query: str, n: int = 5, ctx: "ToolCtx" = None) -> Dict[str, Any]:
    """StackOverflow answers -- the practical half of learning to program."""
    q = urllib.parse.urlencode({"q": query, "sort": "votes", "order": "desc",
                                "site": "stackoverflow", "pagesize": n,
                                "filter": "!nNPvSNVWme"})
    data, status = ctx.get(STACKEX + "?" + q, max_bytes=600_000) if ctx else (None, "no-ctx")
    if data is None:
        return {"questions": [], "status": status}
    try:
        rows = json.loads(data.decode("utf-8", "replace")).get("items", [])
    except Exception as e:
        return {"questions": [], "status": f"parse:{e}"}
    return {"questions": [{"title": htmllib.unescape(r.get("title", "")),
                           "score": r.get("score"), "answers": r.get("answer_count"),
                           "url": r.get("link"),
                           "body": _clean(r.get("body_markdown", ""))[:2000]} for r in rows],
            "count": len(rows), "status": "ok"}


def repos(query: str, n: int = 5, ctx: "ToolCtx" = None) -> Dict[str, Any]:
    """GitHub repository search -- finds implementations, not just prose."""
    data, status = ctx.get(GITHUB + "?" + urllib.parse.urlencode(
        {"q": query, "per_page": n, "sort": "stars"})) if ctx else (None, "no-ctx")
    if data is None:
        return {"repos": [], "status": status}
    try:
        rows = json.loads(data.decode("utf-8", "replace")).get("items", [])
    except Exception as e:
        return {"repos": [], "status": f"parse:{e}"}
    return {"repos": [{"name": r.get("full_name"), "stars": r.get("stargazers_count"),
                       "desc": (r.get("description") or "")[:240], "lang": r.get("language"),
                       "url": r.get("html_url")} for r in rows],
            "count": len(rows), "status": "ok"}


def fetch_page(url: str, ctx: "ToolCtx" = None) -> Dict[str, Any]:
    """Fetch any URL and reduce it to clean text."""
    data, status = ctx.get(url, max_bytes=1_200_000) if ctx else (None, "no-ctx")
    if data is None:
        return {"text": "", "status": status}
    text, title, meta = html_to_text(data, url)
    return {"text": text[:120_000], "title": title, "chars": len(text),
            "meta": meta, "status": "ok"}


def hf_rows(dataset: str, config: str = "default", split: str = "train",
            offset: int = 0, length: int = 50, ctx: "ToolCtx" = None) -> Dict[str, Any]:
    """One page of any HuggingFace dataset, as JSON. This is the slow-drip tap."""
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                "offset": offset, "length": length})
    data, status = ctx.get(HF_ROWS + "?" + q, max_bytes=6_000_000) if ctx else (None, "no-ctx")
    if data is None:
        return {"rows": [], "status": status}
    try:
        d = json.loads(data.decode("utf-8", "replace"))
    except Exception as e:
        return {"rows": [], "status": f"parse:{e}"}
    return {"rows": [r.get("row", {}) for r in d.get("rows", [])],
            "total": d.get("num_rows_total"), "status": "ok"}


def run_python(code: str, timeout: int = 30, ctx: "ToolCtx" = None) -> Dict[str, Any]:
    """Execute Python in a confined child process.

    Containment, not a security boundary: fresh temp working directory, hard
    timeout, capped output, no inheritance of MEGON's own database path, and
    every invocation written to the audit log. That is enough for a system to
    test its own hypotheses and verify its own skills; it is not enough to
    safely run arbitrary hostile code, and it is not claimed to be.
    """
    if ctx is not None and not ctx.policy.allow_exec(code):
        return {"ok": False, "status": "blocked-by-policy"}
    workdir = Path(tempfile.mkdtemp(prefix="megon_run_"))
    script = workdir / "snippet.py"
    script.write_text(code[:200_000], encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if k in ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONHASHSEED", "TMPDIR")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-I", str(script)], cwd=workdir, env=env,
                           capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "")[:20_000]
        err = (p.stderr or "")[-6_000:]
        res = {"ok": p.returncode == 0, "exit": p.returncode, "stdout": out, "stderr": err,
               "secs": round(time.time() - t0, 2), "status": "ok"}
    except subprocess.TimeoutExpired:
        res = {"ok": False, "exit": None, "stdout": "", "stderr": f"timeout after {timeout}s",
               "secs": timeout, "status": "timeout"}
    except Exception as e:
        res = {"ok": False, "exit": None, "stdout": "", "stderr": f"{type(e).__name__}: {e}",
               "secs": round(time.time() - t0, 2), "status": "error"}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if ctx is not None:
        ctx.policy.note(ctx.c, "EXEC" if res["ok"] else "EXEC-FAIL", "run_python",
                        code[:160].replace("\n", " "))
    return res


# ==============================================================================
# the tool context: one object carries config, budget, rate limit, policy, memory
# ==============================================================================
class ToolCtx:
    def __init__(self, cfg: Dict[str, Any], c: sqlite3.Connection,
                 store: Optional[Store] = None, retriever: Optional[Any] = None,
                 answerer: Optional[Any] = None,
                 budget: Optional[Budget] = None, rate: Optional[RateLimiter] = None):
        self.cfg, self.c, self.store = cfg, c, store
        self.retriever, self.answerer = retriever, answerer
        self.budget = budget or Budget(cfg["budget"]["mb_per_day"],
                                       cfg["budget"]["max_docs_per_day"], None, cfg)
        self.rate = rate or RateLimiter(cfg["budget"]["min_request_interval_s"],
                                        cfg["budget"]["max_requests_per_min"])
        self.policy = Policy(cfg)
        self.calls = 0

    def get(self, url: str, max_bytes: Optional[int] = None) -> Tuple[Optional[bytes], str]:
        """The single network funnel: policy, robots, rate limit, budget, audit."""
        self.calls += 1
        if self.budget.exhausted():
            return None, "budget-exhausted"
        data, status = http_get(url, self.cfg, self.rate, self.policy, self.c,
                                max_bytes=max_bytes)
        if data is not None:
            self.budget.charge(len(data))
        return data, status


# ---- memory tools: the agent's own recall -----------------------------------
def remember(text: str, title: str = "", url: str = "", source_id: str = "agent",
             ctx: "ToolCtx" = None) -> Dict[str, Any]:
    if ctx is None or ctx.store is None:
        return {"ok": False, "status": "no-memory"}
    if not ctx.policy.check_content(text, url or "agent", ctx.c):
        return {"ok": False, "status": "blocked-by-policy"}
    doc_id, why = ctx.store.add_doc(url or f"agent://{sha256(text)[:16]}", title, text,
                                    source_id, "agent")
    return {"ok": doc_id is not None, "doc_id": doc_id, "status": why}


def recall(query: str, k: int = 5, ctx: "ToolCtx" = None) -> Dict[str, Any]:
    if ctx is None or ctx.retriever is None:
        return {"hits": [], "status": "no-index"}
    ov = kv_get(ctx.c, "best_overrides", {}) or {}
    return {"hits": ctx.retriever.search(query, k=k, overrides=ov), "status": "ok"}


def think(question: str, ctx: "ToolCtx" = None) -> Dict[str, Any]:
    if ctx is None or ctx.answerer is None:
        return {"answer": "", "status": "no-answerer"}
    ov = kv_get(ctx.c, "best_overrides", {}) or {}
    return ctx.answerer.answer(question, overrides=ov)


# ==============================================================================
# registry -- declared so the agent can reason about what it can do
# ==============================================================================
@dataclass
class Tool:
    name: str
    fn: Callable[..., Dict[str, Any]]
    desc: str
    arg: str
    cost: float          # 0..1, how expensive / slow
    risk: str            # none | net | exec
    learns: str = ""     # what using it tends to improve


TOOLS: List[Tool] = [
    Tool("web_search", lambda a, ctx: web_search(a, 8, ctx),
         "general web search; returns titles, urls, snippets", "query", 0.3, "net",
         "breadth of coverage"),
    Tool("fetch_page", lambda a, ctx: fetch_page(a, ctx),
         "fetch one url and reduce it to clean text", "url", 0.4, "net", "depth on a source"),
    Tool("wikipedia", lambda a, ctx: wikipedia(a, ctx),
         "full plain-text encyclopedia extracts for a topic", "topic", 0.4, "net",
         "stable factual grounding"),
    Tool("arxiv", lambda a, ctx: arxiv(a, 5, ctx),
         "preprint search: title, authors, abstract", "query", 0.4, "net",
         "frontier methods"),
    Tool("papers", lambda a, ctx: papers(a, 5, ctx),
         "OpenAlex scholarly search with citation counts", "query", 0.35, "net",
         "which claims are well supported"),
    Tool("code_search", lambda a, ctx: code_search(a, 5, ctx),
         "top-voted StackOverflow answers", "query", 0.3, "net", "programming skill"),
    Tool("repos", lambda a, ctx: repos(a, 5, ctx),
         "GitHub repository search by stars", "query", 0.3, "net", "working implementations"),
    Tool("hf_rows", lambda a, ctx: hf_rows(*(a.split("|") + ["default", "train", 0, 50])[:1], ctx=ctx),
         "one page of a HuggingFace dataset (dataset|config|split)", "dataset", 0.5, "net",
         "training-grade corpus"),
    Tool("run_python", lambda a, ctx: run_python(a, 30, ctx),
         "execute python in a confined child process and read the result", "code", 0.6, "exec",
         "verified skill acquisition"),
    Tool("remember", lambda a, ctx: remember(a[:200], "", "", "agent", ctx),
         "write a fact or finding into long-term memory", "text", 0.05, "none", "memory"),
    Tool("recall", lambda a, ctx: recall(a, 5, ctx),
         "search long-term memory", "query", 0.05, "none", "recall precision"),
    Tool("think", lambda a, ctx: think(a, ctx),
         "answer from memory with citations", "question", 0.1, "none", "answer quality"),
]

TOOLMAP: Dict[str, Tool] = {t.name: t for t in TOOLS}


def call_tool(name: str, arg: str, ctx: ToolCtx) -> Dict[str, Any]:
    t = TOOLMAP.get(name)
    if t is None:
        return {"ok": False, "status": f"unknown-tool:{name}"}
    try:
        return t.fn(arg, ctx)
    except Exception as e:
        return {"ok": False, "status": f"tool-error:{type(e).__name__}: {e}"}


def tool_catalogue() -> List[Dict[str, Any]]:
    return [{"name": t.name, "desc": t.desc, "arg": t.arg, "cost": t.cost,
             "risk": t.risk, "learns": t.learns} for t in TOOLS]
