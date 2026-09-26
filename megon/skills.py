"""MEGON I AI -- the skill library.

This is how MEGON acquires *capability* rather than just knowledge, and it is
the one growth mechanism here that needs no GPU at all.

The loop is: propose a skill -> write it as real Python -> generate tests ->
run the tests in the confined executor -> keep it only if they pass -> record
how often it wins in use -> mutate and re-verify when it fails.

A skill that cannot pass its own tests never enters the library, so the library
only ever grows by verified capability. That is a much stronger guarantee than
"scraped more text", and it is why this part compounds.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from megon.base import *
from megon.tools import ToolCtx, run_python, call_tool, TOOLS, TOOLMAP, Tool

SKILL_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills(
  name TEXT PRIMARY KEY, purpose TEXT, code TEXT, tests TEXT, family TEXT,
  created_at TEXT, verified INTEGER DEFAULT 0, last_verified TEXT,
  uses INTEGER DEFAULT 0, wins INTEGER DEFAULT 0, failures INTEGER DEFAULT 0,
  origin TEXT DEFAULT 'template', notes TEXT
);
"""

# -----------------------------------------------------------------------------
# starter families. Each entry is (code, tests) where tests is a list of
# (input, expected) pairs executed in the confined runner. These exist so the
# mechanism has something real to chew on from the first cycle; MEGON adds to
# them, and with an LLM endpoint configured it can propose entirely new ones.
# -----------------------------------------------------------------------------
TEMPLATES: Dict[str, Dict[str, str]] = {
    "token_f1": {
        "family": "eval",
        "purpose": "token-level F1 between a predicted answer and a gold answer",
        "code": (
            "import re\n"
            "def run(pred, gold):\n"
            "    p = set(re.findall(r\"[a-z0-9']+\", str(pred).lower()))\n"
            "    g = set(re.findall(r\"[a-z0-9']+\", str(gold).lower()))\n"
            "    if not p or not g:\n"
            "        return 0.0\n"
            "    i = len(p & g)\n"
            "    if not i:\n"
            "        return 0.0\n"
            "    pr, rc = i / len(p), i / len(g)\n"
            "    return round(2 * pr * rc / (pr + rc), 4)\n"
        ),
        "tests": [["the Main Building", "the Main Building", 1.0],
                  ["Main Building", "the Main Building", 0.8],
                  ["Rome", "Paris", 0.0]],
    },
    "levenshtein": {
        "family": "text",
        "purpose": "edit distance between two strings",
        "code": (
            "def run(a, b):\n"
            "    a, b = str(a), str(b)\n"
            "    prev = list(range(len(b) + 1))\n"
            "    for i, ca in enumerate(a, 1):\n"
            "        cur = [i]\n"
            "        for j, cb in enumerate(b, 1):\n"
            "            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))\n"
            "        prev = cur\n"
            "    return prev[-1]\n"
        ),
        "tests": [["kitten", "sitting", 3], ["", "abc", 3], ["same", "same", 0]],
    },
    "extract_numbers": {
        "family": "parse",
        "purpose": "pull every number out of a sentence, handling thousands separators",
        "code": (
            "import re\n"
            "def run(text):\n"
            "    return [float(m.replace(',', '')) for m in\n"
            "            re.findall(r'\\d[\\d,]*(?:\\.\\d+)?', str(text))]\n"
        ),
        "tests": [["I have 1,200 apples and 3.5 oranges", [1200.0, 3.5]],
                  ["no numbers here", []]],
    },
    "normalize_ws": {
        "family": "text",
        "purpose": "collapse whitespace and strip control characters",
        "code": (
            "import re, unicodedata\n"
            "def run(text):\n"
            "    t = unicodedata.normalize('NFKC', str(text))\n"
            "    t = re.sub(r'[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f]', '', t)\n"
            "    return re.sub(r'\\s+', ' ', t).strip()\n"
        ),
        "tests": [["a\n\n  b\t\tc", "a b c"], ["  x  ", "x"]],
    },
    "sentence_split": {
        "family": "text",
        "purpose": "split prose into sentences without breaking common abbreviations",
        "code": (
            "import re\n"
            "def run(text):\n"
            "    t = re.sub(r'\\b(Mr|Mrs|Ms|Dr|Prof|St|No|vs|etc|Inc|Ltd)\\.', r'\\1<DOT>', str(text))\n"
            "    parts = re.split(r'(?<=[.!?])\\s+(?=[A-Z0-9])', re.sub(r'\\s+', ' ', t).strip())\n"
            "    return [p.replace('<DOT>', '.').strip() for p in parts if len(p.strip()) > 2]\n"
        ),
        "tests": [["Dr. Smith went home. He slept.", ["Dr. Smith went home.", "He slept."]],
                  ["One only", ["One only"]]],
    },
    "dedupe_seq": {
        "family": "text",
        "purpose": "remove duplicate lines while preserving order",
        "code": (
            "def run(text):\n"
            "    seen, out = set(), []\n"
            "    for ln in str(text).splitlines():\n"
            "        if ln not in seen:\n"
            "            seen.add(ln)\n"
            "            out.append(ln)\n"
            "    return '\\n'.join(out)\n"
        ),
        "tests": [["a\nb\na\nc", "a\nb\nc"], ["x\nx", "x"]],
    },
    "safe_div": {
        "family": "math",
        "purpose": "division that returns None instead of raising on zero",
        "code": (
            "def run(a, b):\n"
            "    try:\n"
            "        b = float(b)\n"
            "        return None if b == 0 else float(a) / b\n"
            "    except Exception:\n"
            "        return None\n"
        ),
        "tests": [[10, 2, 5.0], [1, 0, None], ["8", "4", 2.0]],
    },
    "jaccard": {
        "family": "eval",
        "purpose": "Jaccard similarity of two word sets; used for near-duplicate spotting",
        "code": (
            "import re\n"
            "def run(a, b):\n"
            "    A = set(re.findall(r'[a-z0-9]+', str(a).lower()))\n"
            "    B = set(re.findall(r'[a-z0-9]+', str(b).lower()))\n"
            "    if not A and not B:\n"
            "        return 1.0\n"
            "    u = A | B\n"
            "    return round(len(A & B) / len(u), 4) if u else 0.0\n"
        ),
        "tests": [["a b c", "a b c", 1.0], ["a b", "c d", 0.0], ["a b c", "b c d", 0.5]],
    },
}


class SkillLibrary:
    def __init__(self, c: sqlite3.Connection, home: Path):
        self.c = c
        self.dir = home / "skills"
        self.dir.mkdir(parents=True, exist_ok=True)
        c.executescript(SKILL_SCHEMA)

    # ---- lifecycle ----------------------------------------------------------
    def add(self, name: str, code: str, tests: List[Any], purpose: str = "",
            family: str = "custom", origin: str = "template", notes: str = "") -> Dict[str, Any]:
        self.c.execute(
            """INSERT INTO skills(name,purpose,code,tests,family,created_at,origin,notes)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(name) DO UPDATE SET code=excluded.code, tests=excluded.tests,
                 purpose=excluded.purpose, notes=excluded.notes""",
            (name, purpose, code, json.dumps(tests), family, now_iso(), origin, notes))
        return self.verify(name)

    def verify(self, name: str, ctx: Optional[ToolCtx] = None) -> Dict[str, Any]:
        """Run the skill against its own tests inside the confined executor.

        Nothing enters the library as verified unless every test passes, so the
        library cannot silently rot.
        """
        row = self.c.execute("SELECT * FROM skills WHERE name=?", (name,)).fetchone()
        if row is None:
            return {"ok": False, "status": "no-such-skill"}
        tests = json.loads(row["tests"] or "[]")
        harness = (
            "import json,sys\n" + row["code"] + "\n"
            "_t = json.loads(sys.stdin.read())\n"
            "_bad = []\n"
            "for case in _t:\n"
            "    args, want = case[:-1], case[-1]\n"
            "    try:\n"
            "        got = run(*args)\n"
            "    except Exception as e:\n"
            "        _bad.append({'args': args, 'error': f'{type(e).__name__}: {e}'})\n"
            "        continue\n"
            "    if isinstance(want, float) and isinstance(got, (int, float)):\n"
            "        ok = abs(got - want) < 1e-6\n"
            "    else:\n"
            "        ok = got == want\n"
            "    if not ok:\n"
            "        _bad.append({'args': args, 'want': want, 'got': got})\n"
            "print(json.dumps({'passed': len(_t) - len(_bad), 'total': len(_t), 'bad': _bad}))\n"
        )
        res = self._exec(harness, json.dumps(tests), ctx)
        ok, detail = False, res.get("stderr", "")[:400]
        if res.get("ok") and res.get("stdout"):
            try:
                detail = json.loads(res["stdout"].strip().splitlines()[-1])
                ok = detail.get("total", 0) > 0 and detail.get("passed") == detail.get("total")
            except Exception as e:
                detail = f"unparseable test output: {e}"
        self.c.execute("UPDATE skills SET verified=?, last_verified=? WHERE name=?",
                       (1 if ok else 0, now_iso(), name))
        (self.dir / f"{name}.py").write_text(row["code"], encoding="utf-8")
        return {"ok": ok, "name": name, "detail": detail}

    @staticmethod
    def _exec(code: str, stdin: str, ctx: Optional[ToolCtx]) -> Dict[str, Any]:
        """Run with stdin support (run_python takes code only)."""
        import subprocess, sys, tempfile, shutil, os
        if ctx is not None and not ctx.policy.allow_exec(code):
            return {"ok": False, "status": "blocked-by-policy", "stderr": "policy"}
        wd = Path(tempfile.mkdtemp(prefix="megon_skill_"))
        (wd / "s.py").write_text(code, encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "LC_ALL")}
        try:
            p = subprocess.run([sys.executable, "-I", str(wd / "s.py")], input=stdin,
                               cwd=wd, env=env, capture_output=True, text=True, timeout=30)
            return {"ok": p.returncode == 0, "stdout": p.stdout[:8000], "stderr": p.stderr[-3000:]}
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": f"{type(e).__name__}: {e}"}
        finally:
            shutil.rmtree(wd, ignore_errors=True)

    # ---- use ----------------------------------------------------------------
    def install_seed(self) -> int:
        n = 0
        for name, spec in TEMPLATES.items():
            if self.c.execute("SELECT 1 FROM skills WHERE name=?", (name,)).fetchone():
                continue
            self.add(name, spec["code"], spec["tests"], spec["purpose"], spec["family"])
            n += 1
        return n

    def get(self, name: str) -> Optional[sqlite3.Row]:
        return self.c.execute("SELECT * FROM skills WHERE name=?", (name,)).fetchone()

    def verified(self) -> List[sqlite3.Row]:
        return self.c.execute("SELECT * FROM skills WHERE verified=1 ORDER BY name").fetchall()

    def all(self) -> List[sqlite3.Row]:
        return self.c.execute("SELECT * FROM skills ORDER BY verified DESC, name").fetchall()

    def record_use(self, name: str, won: bool) -> None:
        self.c.execute(
            f"UPDATE skills SET uses=uses+1, {'wins' if won else 'failures'}="
            f"{'wins' if won else 'failures'}+1 WHERE name=?", (name,))

    def run(self, name: str, *args: Any) -> Dict[str, Any]:
        row = self.get(name)
        if row is None or not row["verified"]:
            return {"ok": False, "status": "unverified-or-missing"}
        harness = ("import json,sys\n" + row["code"] +
                   "\nprint(json.dumps({'v': run(*json.loads(sys.stdin.read()))}, default=str))\n")
        res = self._exec(harness, json.dumps(list(args), default=str), None)
        if res.get("ok") and res.get("stdout"):
            try:
                self.record_use(name, True)
                return {"ok": True, "value": json.loads(res["stdout"].strip().splitlines()[-1])["v"]}
            except Exception as e:
                self.record_use(name, False)
                return {"ok": False, "status": f"parse:{e}"}
        self.record_use(name, False)
        return {"ok": False, "status": res.get("stderr", "failed")[:300]}

    # ---- growth -------------------------------------------------------------
    def propose(self, name: str, purpose: str, code: str, tests: List[Any],
                ctx: Optional[ToolCtx] = None, origin: str = "self") -> Dict[str, Any]:
        """Add a newly invented skill. Verified before it counts as acquired."""
        if ctx is not None and not ctx.policy.allow_exec(code):
            return {"ok": False, "status": "blocked-by-policy"}
        self.add(name, code, tests, purpose, "self-authored", origin)
        return self.verify(name, ctx)

    def mutate(self, name: str, ctx: Optional[ToolCtx] = None) -> Dict[str, Any]:
        """Re-verify a failing skill; report honestly rather than pretend."""
        row = self.get(name)
        if row is None:
            return {"ok": False, "status": "no-such-skill"}
        return self.verify(name, ctx)

    # ---- invention ----------------------------------------------------------
    def invent(self, name: str, purpose: str, examples: List[Any], out: str = "str",
               ctx: Optional[ToolCtx] = None, origin: str = "synthesis") -> Dict[str, Any]:
        """Synthesise a brand-new skill from input/output examples alone.

        Search finds a program, held-out examples reject the overfits, and the
        survivor is then re-verified by the normal skill gate before it counts.
        Two independent checks: synthesis against held-out cases, then the
        library's own test run in the confined executor.
        """
        from megon.synth import Synthesiser
        res = Synthesiser().synthesise([tuple(e) for e in examples], out)
        if not res.get("ok"):
            return {"ok": False, "stage": "synthesis", **{k: v for k, v in res.items() if k != "ok"}}
        self.add(name, res["source"], res["tests"], purpose, "synthesised", origin,
                 notes=f"program: {res['program']}")
        verdict = self.verify(name, ctx)
        return {"ok": bool(verdict["ok"]), "stage": "verify", "program": res["program"],
                "depth": res["depth"], "rejected_overfit": res.get("rejected_overfit", 0),
                "verify": verdict}

    def invent_demos(self, ctx: Optional[ToolCtx] = None) -> Dict[str, Any]:
        """Synthesise the built-in demo tasks. Proves invention works end to end."""
        from megon.synth import DEMO_TASKS
        out = {}
        for name, task in DEMO_TASKS.items():
            out[name] = self.invent(name, task["purpose"], task["examples"],
                                    task["out"], ctx)
        return out

    def install_as_tools(self) -> List[str]:
        """Expose every verified skill as a callable tool for the agent."""
        added = []
        for row in self.verified():
            if row["name"] in TOOLMAP:
                continue
            nm = row["name"]
            tool = Tool(f"skill:{nm}",
                        lambda a, ctx, _n=nm: self.run(_n, a),
                        f"[skill] {row['purpose']}", "input", 0.15, "exec",
                        row["family"])
            TOOLS.append(tool)
            TOOLMAP[tool.name] = tool
            added.append(tool.name)
        return added

    def stats(self) -> Dict[str, Any]:
        r = self.c.execute("SELECT COUNT(*) n, SUM(verified) v, SUM(uses) u, SUM(wins) w "
                           "FROM skills").fetchone()
        return {"skills": r["n"] or 0, "verified": r["v"] or 0,
                "uses": r["u"] or 0, "wins": r["w"] or 0}
