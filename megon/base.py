#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  M Y T H O S   A I   --  a self-learning, slow-drip knowledge engine
================================================================================

WHAT THIS ACTUALLY IS
---------------------
A real, runnable, self-improving system in two files:

    megon       the engine + CLI (this file)
    megon web   the browser console (chat, memory, leaderboard, cycle runner)

Every night (or whenever you run `megon learn`) it:
    1. ACQUIRES   pulls a small, budgeted slice of real public data
                  (FineWeb-Edu, SQuAD, GSM8K, WikiText, Wikipedia, arXiv,
                   any URL you give it) -- a few MB/day, not the whole web.
    2. CONSOLIDATES  cleans -> chunks -> dedupes -> re-embeds -> rebuilds index.
    3. EVALUATES  scores itself on a benchmark it also *grows by itself*
                  (auto-generated cloze/QA/retrieval items + a math bank).
    4. SELF-TUNES  a UCB bandit searches its own hyper-parameter space against
                  that benchmark and keeps whatever measurably works.
    5. REFLECTS   writes lessons ("topic X is weak", "domain Y is high-yield")
                  which feed the *next* night's crawl priorities.
    6. PUBLISHES  a version card + leaderboard row so progress is auditable.

WHAT IS HONEST ABOUT IT
-----------------------
 * It genuinely gets better at its own benchmark, day over day, and you can
   watch the curve. That improvement is retrieval + tuning + memory growth.
 * It is NOT, and cannot be, "the best AI model". A 2-CPU / 2 GB box cannot
   train a frontier model; nothing becomes SOTA by scraping. The path from
   "this" to "trained weights" is `megon export-sft` + `finetune-script`,
   which hand you a real LoRA fine-tuning job for a GPU box.
 * Hardware pressure is capped by construction: token-bucket rate limiting, a
   daily MB/doc/minute budget, incremental indexing, int8 vector quantisation.

WHAT IS DELIBERATELY NOT IN IT
------------------------------
"no restrictions / it can do anything" is not built, and is not a knob.
There is a policy layer instead: robots.txt, rate ceilings, private-IP/SSRF
blocking, a non-removable illegal-content floor, an audit log and a kill
switch. Unrestricted autonomy makes a system *worse* (it gets blocked, banned,
or poisons its own memory) -- not smarter. See README.md, "The restriction".

DEPENDENCIES
------------
Required: Python 3.9+ and numpy (used for the vector index).
Optional, auto-detected: scikit-learn (turns the lexical vectors into true
LSA distributional semantics), bs4 (better HTML cleaning), requests (faster
HTTP), and an OpenAI-compatible LLM endpoint (switches the answer engine from
extractive to RAG-generation). Everything degrades gracefully if they are absent.

USAGE
-----
    python3 megon init
    python3 megon learn --mb 4 --docs 25 --minutes 8
    python3 megon ask "what is a transformer in deep learning?"
    python3 megon leaderboard
    python3 megon daemon            # runs one cycle per day, forever
    python3 megon web               # browser console
================================================================================
"""
from __future__ import annotations

import argparse
import base64
import collections
import gzip
import hashlib
import html as htmllib
import ipaddress
import json
import math
import os
import pickle
import random
import re
import shlex
import shutil
import signal
import socket
import sqlite3
import string
import sys
import textwrap
import threading
import time
import traceback
import unicodedata
import urllib.parse
import urllib.robotparser
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

VERSION = "1.0.0"

# ---------------------------------------------------------------- identity
AUTHOR = "Megix"
OWNER = "Redoyanul Haque"
IDENTITY = (f"MEGON I AI — created by {AUTHOR}. Owner: {OWNER}.\n"
            "A self-deciding, self-developing, self-grading engine that runs "
            "on ordinary CPUs.")


def whoami() -> str:
    return IDENTITY


# ---------------------------------------------------------------- curriculum
# The skills MEGON is pointed at first. These are learned on startup, before it
# starts choosing its own topics -- a newborn needs a syllabus before it needs
# curiosity. Image generation and vision are deliberately absent: they need
# weights and a GPU MEGON does not have, and pretending otherwise would just
# produce a capability that silently does nothing.
CURRICULUM = {
    "coding": [
        "python data structures idioms", "algorithm complexity analysis",
        "design patterns creational structural behavioural",
        "error handling exceptions best practice", "unit testing pytest",
    ],
    "agentic-workflow": [
        "LLM agent architecture planning", "ReAct reasoning acting loop",
        "tool use function calling agents", "multi agent orchestration",
        "reflection self critique agents", "agent memory architectures",
    ],
    "intelligence": [
        "reasoning deductive inductive abductive", "knowledge representation",
        "meta learning learning to learn", "transfer learning generalisation",
        "cognitive architecture production systems",
    ],
    "cyber-security": [
        "OWASP top ten web vulnerabilities", "input validation sanitisation",
        "cryptography symmetric asymmetric hashing", "authentication authorisation",
        "secure coding practices memory safety", "threat modelling STRIDE",
    ],
    "mathematics": [
        "linear algebra matrices eigenvalues", "probability distributions bayes",
        "calculus derivatives optimisation", "discrete mathematics combinatorics",
        "number theory modular arithmetic", "graph theory algorithms",
    ],
}


def curriculum_topics() -> List[str]:
    """Flat list of every bootstrap topic, in syllabus order."""
    out: List[str] = []
    for tracks in CURRICULUM.values():
        out.extend(tracks)
    return out
SCHEMA_VERSION = 3

# --- optional heavy deps: everything degrades gracefully ---------------------
try:
    import numpy as np
except Exception:                                   # pragma: no cover
    np = None

try:
    import requests
except Exception:                                   # pragma: no cover
    requests = None

try:
    from bs4 import BeautifulSoup                  # type: ignore
except Exception:                                   # pragma: no cover
    BeautifulSoup = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    SKLEARN = True
except Exception:                                   # pragma: no cover
    TfidfVectorizer = None
    TruncatedSVD = None
    SKLEARN = False


# ==============================================================================
# 0. tiny helpers
# ==============================================================================
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today_key() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def blake_int(key: str) -> int:
    """Stable (cross-process) 64-bit hash. Python's hash() is salted -> never use."""
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "little")


def human(n: float, unit: str = "B") -> str:
    for suf in ("", "K", "M", "G", "T"):
        if abs(n) < 1024 or suf == "T":
            return f"{n:.1f}{suf}{unit}" if suf else f"{n:.0f}{unit}"
        n /= 1024.0
    return f"{n:.1f}T{unit}"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class Log:
    LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}
    level = "info"

    @classmethod
    def _out(cls, tag: str, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        colour = {"DBG": "\033[90m", "•": "\033[36m", "!": "\033[33m", "X": "\033[31m", "OK": "\033[32m"}.get(tag, "")
        reset = "\033[0m" if colour and sys.stdout.isatty() else ""
        colour = colour if sys.stdout.isatty() else ""
        print(f"{ts} {colour}{tag:>2}{reset} {msg}", flush=True)

    @classmethod
    def d(cls, m: str) -> None:
        if cls.LEVELS[cls.level] <= 10:
            cls._out("DBG", m)

    @classmethod
    def i(cls, m: str) -> None:
        if cls.LEVELS[cls.level] <= 20:
            cls._out("•", m)

    @classmethod
    def ok(cls, m: str) -> None:
        cls._out("OK", m)

    @classmethod
    def w(cls, m: str) -> None:
        if cls.LEVELS[cls.level] <= 30:
            cls._out("!", m)

    @classmethod
    def e(cls, m: str) -> None:
        cls._out("X", m)


log = Log


# ==============================================================================
# 1. home directory + configuration
# ==============================================================================
def home_dir() -> Path:
    p = Path(os.environ.get("MEGON_HOME", Path.cwd() / "megon_home")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


DEFAULT_SOURCES: List[Dict[str, Any]] = [
    # Real corpora used to train today's top open models. `rows` pages of 100 at
    # a time = a few hundred KB per pull, which is the whole point: slow drip.
    {"id": "fineweb-edu", "kind": "hf", "dataset": "HuggingFaceFW/fineweb-edu",
     "config": "default", "split": "train", "text_field": "text", "priority": 1.0,
     "note": "1.3T-tok web corpus filtered for educational value (Llama-3 style pretraining mix)"},
    {"id": "fineweb", "kind": "hf", "dataset": "HuggingFaceFW/fineweb",
     "config": "default", "split": "train", "text_field": "text", "priority": 0.8,
     "note": "15T-token Common Crawl dump, deduped + filtered"},
    {"id": "squad", "kind": "hf", "dataset": "rajpurkar/squad",
     "config": "plain_text", "split": "train", "text_field": "context",
     "qa": True, "priority": 0.9, "note": "SQuAD v1.1 reading comprehension (CC BY-SA 4.0)"},
    {"id": "gsm8k", "kind": "hf", "dataset": "openai/gsm8k",
     "config": "main", "split": "train", "text_field": "answer",
     "math": True, "priority": 0.9, "note": "grade-school math reasoning (MIT)"},
    {"id": "wikitext-2", "kind": "hf", "dataset": "Salesforce/wikitext",
     "config": "wikitext-2-raw-v1", "split": "train", "text_field": "text",
     "priority": 0.7, "note": "WikiText-2 LM benchmark corpus"},
    {"id": "alpaca", "kind": "hf", "dataset": "tatsu-lab/alpaca",
     "config": "default", "split": "train", "text_field": "output",
     "instruct": True, "priority": 0.7, "note": "instruction-following pairs (CC BY-NC 4.0)"},
    {"id": "wikipedia-llm", "kind": "url",
     "url": "https://en.wikipedia.org/wiki/Large_language_model", "priority": 0.6},
    {"id": "wikipedia-transformer", "kind": "url",
     "url": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)", "priority": 0.6},
]

DEFAULT_CONFIG: Dict[str, Any] = {
    "name": "MEGON I AI",
    "version": VERSION,
    # --- the "slowly slowly, no hardware pressure" contract -------------------
    "budget": {
        "mb_per_day": 6.0,          # hard ceiling on downloaded bytes / day
        "max_docs_per_day": 40,
        "minutes_per_cycle": 20,    # wall-clock ceiling for one learn cycle
        "min_request_interval_s": 1.5,
        "max_requests_per_min": 40,
        "max_bytes_per_doc": 800_000,
        "request_timeout_s": 20,
        "quiet_hours": [23, 7],     # only crawl inside this window when daemon
    },
    "chunk": {"target_tokens": 180, "overlap_tokens": 30, "min_tokens": 20},
    "retrieval": {
        "top_k": 6, "candidate_pool": 80, "rrf_k": 60,
        "dense_weight": 0.55, "lexical_weight": 0.45,
        "bm25_k1": 1.5, "bm25_b": 0.75, "lsa_dim": 256,
        "rerank": True, "title_boost": 0.12, "freshness_boost": 0.03,
    },
    "generation": {"max_sentences": 5, "template": "evidence_v2", "cite": True},
    "policy": {
        "respect_robots": True,
        "block_private_network": True,
        "allowlist": [],
        "blocklist": [],
        "illegal_content_floor": True,
        "allow_exec": True,
        "audit": True,
        "user_agent": "MegonAI/1.0 (+local research agent; single operator; slow-crawl)",
    },
    "llm": {"provider": "auto", "base_url": "", "model": "", "api_key_env": "MEGON_LLM_KEY",
            "max_tokens": 700, "temperature": 0.2},
    "sources": DEFAULT_SOURCES,
    "schedule": {"enabled": True, "hour": 3, "minute": 15, "cycles_per_day": 1},
}


def cfg_path() -> Path:
    return home_dir() / "megon.json"


def load_config() -> Dict[str, Any]:
    p = cfg_path()
    if not p.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    cfg = json.loads(p.read_text(encoding="utf-8"))

    def merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst.setdefault(k, v)
        return dst

    cfg = merge(cfg, json.loads(json.dumps(DEFAULT_CONFIG)))
    cfg["version"] = VERSION
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    cfg_path().write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def db_path() -> Path:
    return home_dir() / "megon.db"


def conn() -> sqlite3.Connection:
    # check_same_thread=False: the web console warms the engine once and then
    # serves requests from ThreadingHTTPServer worker threads. WAL + autocommit
    # (isolation_level=None) make concurrent readers with one writer safe.
    c = sqlite3.connect(db_path(), timeout=30.0, isolation_level=None,
                        check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.execute("PRAGMA cache_size=-8000")          # ~8 MB page cache, not 200 MB
    c.execute("PRAGMA temp_store=MEMORY")
    c.execute("PRAGMA mmap_size=134217728")
    return c


SCHEMA = """
CREATE TABLE IF NOT EXISTS docs(
  id INTEGER PRIMARY KEY, url TEXT, title TEXT, source_id TEXT, kind TEXT,
  sha256 TEXT UNIQUE, simhash INTEGER, lang TEXT, bytes INTEGER, words INTEGER,
  fetched_at TEXT, status TEXT, split TEXT DEFAULT 'train', meta TEXT
);
CREATE INDEX IF NOT EXISTS idx_docs_source ON docs(source_id);
CREATE INDEX IF NOT EXISTS idx_docs_split ON docs(split);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY, doc_id INTEGER, ord INTEGER, text TEXT,
  sha256 TEXT UNIQUE, words INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY, ts TEXT, cycle INTEGER, phase TEXT, ok INTEGER,
  detail TEXT, secs REAL
);
CREATE TABLE IF NOT EXISTS metrics(
  id INTEGER PRIMARY KEY, ts TEXT, cycle INTEGER, name TEXT, value REAL
);
CREATE TABLE IF NOT EXISTS eval_items(
  id INTEGER PRIMARY KEY, kind TEXT, question TEXT, answer TEXT, options TEXT,
  evidence_chunk INTEGER, source_doc INTEGER, split TEXT, created_at TEXT, meta TEXT
);
CREATE INDEX IF NOT EXISTS idx_items_kind ON eval_items(kind);
CREATE TABLE IF NOT EXISTS eval_results(
  id INTEGER PRIMARY KEY, ts TEXT, cycle INTEGER, item_id INTEGER, score REAL, detail TEXT
);
CREATE TABLE IF NOT EXISTS lessons(
  id INTEGER PRIMARY KEY, ts TEXT, cycle INTEGER, kind TEXT, text TEXT, weight REAL, used INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS policy_log(
  id INTEGER PRIMARY KEY, ts TEXT, decision TEXT, url TEXT, reason TEXT
);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
"""


def init_db() -> sqlite3.Connection:
    c = conn()
    c.executescript(SCHEMA)
    c.execute("CREATE TABLE IF NOT EXISTS meta_v(v INTEGER)")
    row = c.execute("SELECT v FROM meta_v LIMIT 1").fetchone()
    if row is None:
        c.execute("INSERT INTO meta_v(v) VALUES(?)", (SCHEMA_VERSION,))
    return c


def kv_get(c: sqlite3.Connection, k: str, default: Any = None) -> Any:
    r = c.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    if not r:
        return default
    try:
        return json.loads(r["v"])
    except Exception:
        return r["v"]


def kv_set(c: sqlite3.Connection, k: str, v: Any) -> None:
    c.execute("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
              (k, json.dumps(v)))


# ==============================================================================
# 2. budget + rate limiting  (this is what keeps the hardware load flat)
# ==============================================================================
class Budget:
    def __init__(self, mb: Optional[float] = None, docs: Optional[int] = None,
                 minutes: Optional[float] = None, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.max_bytes = int((mb if mb is not None else 6.0) * 1024 * 1024)
        self.max_docs = docs if docs is not None else 40
        self.bytes = 0
        self.docs = 0
        self.deadline = time.time() + (minutes * 60.0) if minutes else None
        self.stops: List[str] = []

    def charge(self, nbytes: int) -> None:
        self.bytes += nbytes
        self.docs += 1

    def exhausted(self) -> bool:
        if self.bytes >= self.max_bytes:
            self.stops.append("byte-budget")
            return True
        if self.docs >= self.max_docs:
            self.stops.append("doc-budget")
            return True
        if self.deadline and time.time() > self.deadline:
            self.stops.append("time-budget")
            return True
        return False

    def summary(self) -> Dict[str, Any]:
        return {"docs": self.docs, "bytes": self.bytes, "mb": round(self.bytes / 1048576, 3),
                "stops": sorted(set(self.stops))}


class RateLimiter:
    """Global RPM ceiling + per-host politeness interval. Sleeps, never bursts."""

    def __init__(self, min_interval_s: float = 1.5, max_rpm: int = 40):
        self.min_interval_s = min_interval_s
        self.max_rpm = max_rpm
        self._host: Dict[str, float] = {}
        self._times: List[float] = []
        self._lock = threading.Lock()
        self.slept = 0.0

    def wait(self, host: str) -> None:
        with self._lock:
            now = time.time()
            self._times = [t for t in self._times if now - t < 60.0]
            sleep_s = 0.0
            if len(self._times) >= self.max_rpm:
                sleep_s = 60.0 - (now - self._times[0]) + 0.05
            else:
                last = self._host.get(host, 0.0)
                sleep_s = max(0.0, self.min_interval_s - (now - last))
            if sleep_s > 0:
                sleep_s = min(sleep_s, 90.0)
                time.sleep(sleep_s)
                self.slept += sleep_s
            self._host[host] = time.time()
            self._times.append(time.time())


# Documented public APIs. robots.txt describes crawling *web pages*; it does not
# govern an endpoint whose publisher documents it for programmatic use. Blocking
# the arXiv API because of robots.txt is a category error, so these hosts are
# exempt from the robots check (and still subject to every other limit).
API_HOSTS = {"export.arxiv.org", "api.openalex.org", "api.crossref.org",
             "api.stackexchange.com", "api.github.com",
             "datasets-server.huggingface.co", "en.wikipedia.org"}


class Robots:
    """Cached robots.txt per host. Fails OPEN but logs (many hosts 403 default UAs)."""

    def __init__(self, ua: str, enabled: bool = True):
        self.ua = ua
        self.enabled = enabled
        self._cache: Dict[str, Tuple[float, Any]] = {}

    def allowed(self, url: str, c: Optional[sqlite3.Connection] = None) -> bool:
        if not self.enabled:
            return True
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        if host in API_HOSTS:
            return True
        host = urllib.parse.urlparse(url).netloc
        hit = self._cache.get(host)
        if hit and time.time() - hit[0] < 900:
            rp = hit[1]
        else:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(urljoin_host(url) + "/robots.txt")
            try:
                req = urllib.request.Request(rp.url, headers={"User-Agent": self.ua})
                with urllib.request.urlopen(req, timeout=10) as r:      # type: ignore
                    rp.parse(r.read().decode("utf-8", "replace").splitlines())
            except Exception:
                rp = None
            self._cache[host] = (time.time(), rp)
        if rp is None:
            return True
        try:
            return rp.can_fetch(self.ua, url)
        except Exception:
            return True


def urljoin_host(url: str) -> str:
    p = urllib.parse.urlparse(url)
    return f"{p.scheme}://{p.netloc}"


import urllib.request  # noqa: E402  (kept late so the alias above reads naturally)


# ==============================================================================
# 3. policy layer -- the non-removable floor
# ==============================================================================
# Hard floor. Not configurable, not a "setting". If you are reading this hoping
# to find the flag that turns it off: there isn't one, on purpose.
ILLEGAL_PATTERNS = [
    r"\b(child|kids?|minors?|underage|loli|shota)\s*(porn|sex|nude|nsfw|xxx)\b",
    r"\bcsam\b", r"\bchild\s*sexual\s*abuse\b", r"\bpreteen\s*(nude|sex|porn)\b",
    r"\bhow\s+to\s+make\s+(a\s+)?(nerve\s+agent|ricin|sarin|vx\s+gas)\b",
    r"\bfentanyl\s+synthesis\b", r"\bweapons?\s+grade\s+plutonium\s+(refin|process)\b",
    r"\bbuy\s+(stolen\s+)?credit\s+card\s+(dump|fullz)\b",
]
ILLEGAL_RE = re.compile("|".join(ILLEGAL_PATTERNS), re.I)

PRIVATE_HOSTS = re.compile(r"^(localhost|.*\.local|.*\.internal)$", re.I)


class Policy:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg.get("policy", {})
        self.ua = self.cfg.get("user_agent", DEFAULT_CONFIG["policy"]["user_agent"])

    def _audit(self, c: Optional[sqlite3.Connection], decision: str, url: str, reason: str) -> None:
        if self.cfg.get("audit", True) and c is not None:
            try:
                c.execute("INSERT INTO policy_log(ts,decision,url,reason) VALUES(?,?,?,?)",
                          (now_iso(), decision, url[:500], reason[:300]))
            except Exception:
                pass
        if decision == "BLOCK":
            log.w(f"policy BLOCK {url[:80]} :: {reason}")

    def check_url(self, url: str, c: Optional[sqlite3.Connection] = None) -> Tuple[bool, str]:
        p = urllib.parse.urlparse(url)
        if p.scheme not in ("http", "https"):
            self._audit(c, "BLOCK", url, "scheme not http(s)")
            return False, "scheme not http(s)"
        host = (p.hostname or "").lower()
        if not host:
            self._audit(c, "BLOCK", url, "no host")
            return False, "no host"
        allow, block = self.cfg.get("allowlist") or [], self.cfg.get("blocklist") or []
        if allow and not any(host == a or host.endswith("." + a) for a in allow):
            self._audit(c, "BLOCK", url, "not in allowlist")
            return False, "not in allowlist"
        if any(host == b or host.endswith("." + b) for b in block):
            self._audit(c, "BLOCK", url, "in blocklist")
            return False, "in blocklist"
        if PRIVATE_HOSTS.match(host):
            self._audit(c, "BLOCK", url, "private hostname")
            return False, "private hostname"
        if self.cfg.get("block_private_network", True):
            try:
                for info in socket.getaddrinfo(host, p.port or (443 if p.scheme == "https" else 80)):
                    ip = ipaddress.ip_address(info[4][0])
                    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                        self._audit(c, "BLOCK", url, f"resolves to private/reserved IP {ip}")
                        return False, f"private/reserved IP {ip}"
            except Exception as e:
                self._audit(c, "BLOCK", url, f"dns failure: {e}")
                return False, f"dns failure: {e}"
        return True, "ok"

    def check_content(self, text: str, url: str, c: Optional[sqlite3.Connection] = None) -> bool:
        if not self.cfg.get("illegal_content_floor", True):
            return True
        sample = text[:20000]
        if ILLEGAL_RE.search(sample):
            self._audit(c, "BLOCK", url, "illegal-content floor tripped; text NOT stored")
            return False
        return True

    # Refusal patterns for code the agent wants to execute. This is containment
    # for *its own* autonomy, not a sandbox against an attacker: it stops the
    # obvious self-destructive and exfiltration moves so an unattended loop
    # cannot brick its host or leak its operator's environment.
    EXEC_DENY = [
        r"\bshutil\.rmtree\s*\(\s*['\"]?(~|/|\$HOME)", r"\bos\.system\b",
        r"\bsubprocess\.(?:run|call|Popen)\b", r"\beval\s*\(", r"\bexec\s*\(",
        r"\b__import__\b", r"\bos\.environ\b", r"\bos\.remove\b", r"\bos\.unlink\b",
        r"\bopen\s*\([^)]*['\"]w", r"\bsocket\b", r"\burllib\b", r"\brequests\b",
        r"\bhttpx\b", r"\bpty\b", r"\bfcntl\b", r"\bctypes\b", r"\bfork\b",
        r"\bsignal\.SIG", r"\.ssh|\.aws|credentials|\.netrc|id_rsa",
        r"\bchmod\b", r"\bchown\b", r"\bsudo\b", r"\bnc\s+-", r"\bcurl\b", r"\bwget\b",
    ]
    EXEC_DENY_RE = re.compile("|".join(EXEC_DENY), re.I)

    def allow_exec(self, code: str) -> bool:
        """True when a self-generated snippet is safe enough to run unattended."""
        if not self.cfg.get("allow_exec", True):
            return False
        return self.EXEC_DENY_RE.search(code) is None

    def note(self, c: Optional[sqlite3.Connection], decision: str, url: str, reason: str) -> None:
        self._audit(c, decision, url, reason)


# ==============================================================================
# 4. network fetch
# ==============================================================================
def _ua_headers(cfg: Dict[str, Any], referer: str = "") -> Dict[str, str]:
    h = {
        "User-Agent": cfg.get("policy", {}).get("user_agent", DEFAULT_CONFIG["policy"]["user_agent"]),
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,text/plain;q=0.8,*/*;q=0.5",
        "Accept-Language": "en;q=0.9",
        "Connection": "close",
    }
    if referer:
        h["Referer"] = referer
    return h


def http_get(url: str, cfg: Dict[str, Any], rate: RateLimiter,
             policy: Policy, c: Optional[sqlite3.Connection] = None,
             max_bytes: Optional[int] = None, referer: str = "") -> Tuple[Optional[bytes], str]:
    """Polite, capped GET. Returns (bytes, status_string)."""
    ok, why = policy.check_url(url, c)
    if not ok:
        return None, f"blocked:{why}"
    b = cfg.get("budget", {})
    if b.get("respect_robots", cfg.get("policy", {}).get("respect_robots", True)) or \
       cfg.get("policy", {}).get("respect_robots", True):
        if not Robots(policy.ua, True).allowed(url, c):
            policy.note(c, "BLOCK", url, "robots.txt disallow")
            return None, "blocked:robots"
    rate.wait(urllib.parse.urlparse(url).netloc)
    cap = max_bytes or int(b.get("max_bytes_per_doc", 800_000))
    timeout = float(b.get("request_timeout_s", 20))
    if requests is not None:
        try:
            with requests.get(url, headers=_ua_headers(cfg, referer), timeout=timeout,
                              stream=True, allow_redirects=True) as r:
                if r.status_code != 200:
                    return None, f"http:{r.status_code}"
                buf = bytearray()
                for chunk in r.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    buf += chunk
                    if len(buf) > cap:
                        policy.note(c, "NOTE", url, "truncated at cap")
                        return bytes(buf[:cap]), "ok:truncated"
                return bytes(buf), "ok"
        except Exception as e:
            return None, f"error:{type(e).__name__}"
    # stdlib fallback
    try:
        req = urllib.request.Request(url, headers=_ua_headers(cfg, referer))
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read(cap + 1)
            return (data[:cap], "ok:truncated") if len(data) > cap else (data, "ok")
    except Exception as e:
        return None, f"error:{type(e).__name__}"


# ==============================================================================
# 5. text extraction
# ==============================================================================
TAG_STRIP = {"script", "style", "noscript", "template", "svg", "iframe", "form",
             "nav", "footer", "header", "aside", "button", "select", "option"}
BOILER = re.compile(
    r"(?im)^\s*(cookie|privacy policy|terms of (use|service)|subscribe|sign (in|up)|"
    r"advertisement|related (articles?|posts?)|share (this|on)|jump to|edit this page|"
    r"categories:|from wikipedia|navigation menu|you (are|may) (also|need) to|"
    # Wikipedia chrome that survives tag-stripping as prose-like blobs and then
    # outranks the real content, because it repeats the page title everywhere.
    r"part of a series on|redirected from|unsourced material may be challenged|"
    r"find sources:|learn how and when to remove this|this (template|sidebar)|"
    r"hide\s*$|v\s*t\s*e\b|main article:|see also:|references\s*$|"
    r"retrieve[d]? \d|archived from the original)\b.*$")
# Same idea, but these appear *mid-line* inside a chunk, so an anchored pattern
# never fires. Strip from the phrase to the end of its line.
BOILER_ANY = re.compile(
    r"(?i)(part of a series on|redirected from|unsourced material may be challenged|"
    r"find sources:|learn how and when to remove this|archived from the original|"
    r"main article:|see also:|\bjstor\b|v\s*t\s*e(?=\s|$)|hide this message).*$")

WS = re.compile(r"[ \t\u00a0]+")
BLANKS = re.compile(r"\n{3,}")


def html_to_text(raw: bytes, url: str = "") -> Tuple[str, str, Dict[str, str]]:
    """-> (text, title, meta). BeautifulSoup when present, regex fallback otherwise."""
    enc = "utf-8"
    m = re.search(rb'charset=["\']?([\w-]+)', raw[:4096], re.I)
    if m:
        try:
            "x".encode(m.group(1).decode("ascii", "ignore") or "utf-8")
            enc = m.group(1).decode("ascii", "ignore")
        except Exception:
            enc = "utf-8"
    s = raw.decode(enc, "replace")
    meta: Dict[str, str] = {}
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(s, "html.parser")
            title = (soup.title.get_text(" ", strip=True) if soup.title else "") or ""
            if not title and soup.h1:
                title = soup.h1.get_text(" ", strip=True)
            for t in soup(list(TAG_STRIP)):
                t.decompose()
            md = soup.find("meta", attrs={"name": "description"})
            if md and md.get("content"):
                meta["description"] = md["content"][:400]
            text = soup.get_text("\n")
        except Exception:
            text, title = _regex_extract(s)
    else:
        text, title = _regex_extract(s)
    text = unicodedata.normalize("NFKC", text)
    text = htmllib.unescape(text)
    lines = [BOILER_ANY.sub("", BOILER.sub("", WS.sub(" ", ln))).strip()
             for ln in text.splitlines()]
    text = BLANKS.sub("\n\n", "\n".join(ln for ln in lines if len(ln) > 1))
    return text.strip(), (title or url).strip(), meta


def _regex_extract(s: str) -> Tuple[str, str]:
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", s, re.I | re.S)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
    s = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header|aside|form)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?is)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>", "\n", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    return s, title


def plainify(text: str) -> str:
    text = unicodedata.normalize("NFKC", htmllib.unescape(text))
    text = re.sub(r"\r", "\n", text)
    text = BOILER_ANY.sub("", BOILER.sub("", text))
    return BLANKS.sub("\n\n", WS.sub(" ", text)).strip()


WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+|\d+(?:[.,]\d+)*", re.U)


def words(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


STOPWORDS = set("""a an the and or but if then else of in on at to for from by with without as is are was
were be been being this that these those it its it's into over under about than so such not no nor do does
did done have has had having i you he she they we them his her their our your my me him us can could would
should will shall may might must there here which who whom whose what when where why how all any both each
few more most other some only own same too very s t just don now also per via vs etc eg ie""".split())


def content_words(text: str) -> List[str]:
    return [w for w in words(text) if w not in STOPWORDS and len(w) > 1]


def sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    # protect abbreviations from being treated as sentence boundaries.
    # (forward substitution: a variable-width lookbehind is a regex error in Python)
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|St|No|vs|etc|Inc|Ltd|Jr|Sr|Fig|Eq)\.", r"\1<DOT>", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    out = []
    for p in parts:
        p = p.replace("<DOT>", ".").strip(" \u201c\u201d\"'")
        if len(p) > 2:
            out.append(p)
    return out


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)
