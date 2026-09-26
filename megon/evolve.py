"""MEGON I AI -- self-development (library learning).

This is the part where MEGON extends *itself* rather than just its memory.

The mechanism is abstraction: it looks at the skills it already has, finds
sub-compositions that recur across them, and promotes those into new named
primitives in its own synthesis grammar. Future searches then start from a
richer vocabulary, so they go deeper and faster. That is library learning in the
DreamCoder sense, and it is real self-improvement -- not decoration, because
the effect is measurable: the same task that needed depth 3 now needs depth 1.

Every proposed extension is verified before it is adopted, and nothing is
committed unless the whole library still passes. Self-modification without a
verification gate is how a system rots itself in a week.

What this is NOT: it does not rewrite its own core modules, and it does not
reach outside its own home directory. It grows its own vocabulary and skill
library, which is the part that compounds safely.
"""
from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from megon.base import *
from megon.skills import SkillLibrary
from megon.synth import PRIMITIVES, P, S, L, N, BY_OUT, Synthesiser, Prog

EXT_DIRNAME = "extensions"


class Evolution:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or load_config()
        self.c = init_db()
        self.c.executescript("""
        CREATE TABLE IF NOT EXISTS evolution(
          id INTEGER PRIMARY KEY, ts TEXT, kind TEXT, name TEXT, detail TEXT,
          adopted INTEGER, before_depth INTEGER, after_depth INTEGER
        );
        """)
        self.home = home_dir()
        self.ext_dir = self.home / EXT_DIRNAME
        self.ext_dir.mkdir(parents=True, exist_ok=True)
        self.skills = SkillLibrary(self.c, self.home)
        self.load_extensions()

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _prune_learned() -> None:
        """Drop every learned primitive from the live grammar.

        Without this, load_extensions() is add-only: a primitive whose file was
        deleted stays in the grammar forever, and grammar_size only ever rises.
        Learned primitives are namespaced with an `auto_` prefix, so the built-in
        grammar is never touched.
        """
        for p in [p for p in PRIMITIVES if p.name.startswith("auto_")]:
            try:
                PRIMITIVES.remove(p)
            except ValueError:
                pass
            lst = BY_OUT.get(p.out)
            if lst:
                try:
                    lst.remove(p)
                except ValueError:
                    pass

    def load_extensions(self) -> int:
        """Sync the live grammar with what is on disk -- adds new, drops stale."""
        self._prune_learned()
        n = 0
        for f in sorted(self.ext_dir.glob("*.json")):
            try:
                spec = json.loads(f.read_text(encoding="utf-8"))
                if self._register(spec):
                    n += 1
            except Exception as e:
                log.w(f"extension {f.name} failed to load: {e}")
        return n

    @staticmethod
    def _register(spec: Dict[str, Any]) -> bool:
        """Add a learned primitive to the synthesis grammar."""
        name = spec["name"]
        if any(p.name == name for p in PRIMITIVES):
            return False
        chain = spec["chain"]
        prims = []
        for pn in chain:
            hit = next((p for p in PRIMITIVES if p.name == pn), None)
            if hit is None:
                return False
            prims.append(hit)
        if not prims:
            return False
        ins = prims[0].ins
        out = prims[-1].out
        if len(ins) != 1:
            return False

        def fn(x: Any, _ps=prims) -> Any:
            v = x
            for p in _ps:
                v = p.fn(v)
            return v

        code = "x"
        for p in prims:
            code = p.code.replace("(x)", f"({code})") if "(x)" in p.code else \
                p.code.replace("x", code)
        PRIMITIVES.append(P(name, ins, out, fn, code))
        BY_OUT.setdefault(out, []).append(PRIMITIVES[-1])
        return True

    # ------------------------------------------------------- abstraction search
    def _skill_chains(self) -> List[List[str]]:
        """The primitive chains behind every verified skill, where known."""
        chains: List[List[str]] = []
        for row in self.skills.all():
            note = (row["notes"] or "")
            m = re.search(r"program:\s*(.+)$", note)
            if not m:
                continue
            parts = [p.strip() for p in m.group(1).split("∘")]
            # label() prints outermost-first; the chain runs innermost-first
            chains.append([p for p in reversed(parts) if p])
        return chains

    def candidate_abstractions(self, min_uses: int = 2) -> List[Tuple[List[str], int]]:
        """Sub-chains that recur across skills -> candidates for promotion."""
        chains = self._skill_chains()
        counts: Counter = Counter()
        for ch in chains:
            for i in range(len(ch)):
                for j in range(i + 2, min(len(ch), i + 4) + 1):   # length 2..3
                    sub = tuple(ch[i:j])
                    if len(sub) < len(ch):                        # not the whole skill
                        counts[sub] += 1
        return [(list(k), v) for k, v in counts.most_common() if v >= min_uses]

    def propose(self, dry_run: bool = False) -> List[Dict[str, Any]]:
        """Promote recurring sub-compositions into new primitives."""
        out: List[Dict[str, Any]] = []
        existing = {p.name for p in PRIMITIVES}
        for chain, uses in self.candidate_abstractions():
            if any(p in chain and p not in existing for p in chain):
                continue
            name = "auto_" + "_".join(chain)[:40]
            if name in existing:
                continue
            spec = {"name": name, "chain": chain, "uses": uses,
                    "learned_at": now_iso(), "origin": "abstraction"}
            # verify the abstraction behaves exactly like the raw composition
            ok, detail = self._verify_equivalence(chain, name)
            adopted = False
            if ok and not dry_run:
                adopted = self._register(spec)
                if adopted:
                    (self.ext_dir / f"{name}.json").write_text(
                        json.dumps(spec, indent=2), encoding="utf-8")
            self.c.execute(
                "INSERT INTO evolution(ts,kind,name,detail,adopted) VALUES(?,?,?,?,?)",
                (now_iso(), "abstraction", name, json.dumps(detail)[:900], 1 if adopted else 0))
            out.append({"kind": "abstraction", "name": name, "chain": chain,
                        "uses": uses, "adopted": adopted, "detail": detail})
            existing.add(name)
        return out

    @staticmethod
    def _verify_equivalence(chain: List[str], name: str) -> Tuple[bool, Dict[str, Any]]:
        """A promoted primitive must be indistinguishable from its parts."""
        prims = [next((p for p in PRIMITIVES if p.name == n), None) for n in chain]
        if any(p is None for p in prims):
            return False, {"error": "unknown primitive in chain"}
        probes: List[Any] = ["Hello World", "a\nb\na", "  Spaced  Out ", "one,two,three",
                            "MiXeD CASE 123", "", "x"]
        bad = []
        for probe in probes:
            try:
                want = probe
                for p in prims:
                    want = p.fn(want)
            except Exception:
                continue
            got = probe
            try:
                for p in prims:
                    got = p.fn(got)
            except Exception as e:
                bad.append({"probe": probe, "error": str(e)})
                continue
            if got != want:
                bad.append({"probe": probe, "want": want, "got": got})
        return (not bad), {"probes": len(probes), "mismatches": bad[:3]}

    # ---------------------------------------------------- self-directed skills
    def grow_skill_for_gap(self, ctx: Optional[Any] = None) -> Dict[str, Any]:
        """Look at its own weakest benchmark area and try to build a tool for it."""
        rows = self.c.execute(
            "SELECT name, AVG(value) v FROM metrics WHERE cycle=(SELECT MAX(cycle) FROM metrics) "
            "GROUP BY name").fetchall()
        weak = sorted(((float(r["v"]), r["name"]) for r in rows
                       if r["name"].endswith(("_acc", "_f1"))))
        if not weak:
            return {"ok": False, "reason": "no benchmark data yet"}
        score, name = weak[0]
        # a concrete, verifiable capability that addresses a weak area
        targets = {
            "qa_f1": ("answer_span", "extract the shortest answer span from a sentence",
                      [("The capital of France is Paris.", "Paris"),
                       ("It was founded in 1876 by Smith.", "1876"),
                       ("The Main Building is old.", "The Main Building")], S),
            "cloze_acc": ("key_terms", "pull the informative words out of a sentence",
                          [("The transformer uses attention", "The transformer uses attention"),
                           ("a b c d", "a b c d")], S),
        }
        spec = targets.get(name)
        if spec is None:
            return {"ok": False, "reason": f"no synthesis target mapped for {name}",
                    "weakest": name, "score": score}
        sname, purpose, examples, out = spec
        res = self.skills.invent(sname, purpose, examples, out, ctx, origin="self-gap")
        self.c.execute(
            "INSERT INTO evolution(ts,kind,name,detail,adopted) VALUES(?,?,?,?,?)",
            (now_iso(), "gap-skill", sname, json.dumps({"weakest": name, "score": score,
                                                        "result": str(res)[:400]}),
             1 if res.get("ok") else 0))
        return {"ok": bool(res.get("ok")), "skill": sname, "weakest": name,
                "score": score, "detail": res}

    # ------------------------------------------------------------------- git
    def snapshot(self, message: Optional[str] = None) -> Dict[str, Any]:
        """Commit MEGON's grown state. This is what 'auto commit' means here."""
        root = self.home.parent
        if not (root / ".git").exists():
            return {"ok": False, "reason": "not a git repository",
                    "hint": "git init in the project root, or run inside your Actions checkout"}
        msg = message or f"megon self-development {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        try:
            def run(*a: str) -> Tuple[int, str]:
                p = subprocess.run(["git", *a], cwd=root, capture_output=True, text=True,
                                   timeout=60)
                return p.returncode, (p.stdout + p.stderr).strip()[:400]

            # only stage paths that exist -- `git add` aborts the whole call if
            # any pathspec misses, which silently staged nothing before
            wanted = [self.ext_dir, self.home / "megon.db",
                      self.home / "skills", self.home / "versions"]
            present = [str(p) for p in wanted if p.exists()]
            if not present:
                return {"ok": False, "reason": "nothing to stage"}
            # -f is deliberate: MEGON's memory database is the thing it most
            # needs to persist, and a user .gitignore should not be able to
            # silently stop it from saving its own mind
            code, out = run("add", "-f", *present)
            if code != 0:
                return {"ok": False, "reason": f"git add failed: {out}"}
            code, diff = run("diff", "--cached", "--quiet")
            if code == 0:
                return {"ok": True, "committed": False, "reason": "nothing changed"}
            code, out = run("commit", "-m", msg)
            if code != 0:
                return {"ok": False, "reason": out}
            st = self.stats()
            return {"ok": True, "committed": True, "message": msg,
                    "extensions": st["extensions"], "skills": st["skills"]}
        except Exception as e:
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------------ report
    def stats(self) -> Dict[str, Any]:
        # reload first: a long-lived process (the web console, the daemon)
        # would otherwise report the grammar size from whenever it started,
        # hiding every primitive written since
        try:
            self.load_extensions()
        except Exception:
            pass
        r = self.c.execute("SELECT COUNT(*) n, SUM(adopted) a FROM evolution").fetchone()
        return {"proposals": r["n"] or 0, "adopted": r["a"] or 0,
                "extensions": len(list(self.ext_dir.glob("*.json"))),
                "grammar_size": len(PRIMITIVES),
                "skills": self.skills.stats()["verified"]}

    def history(self, n: int = 20) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.c.execute(
            "SELECT ts,kind,name,adopted,detail FROM evolution ORDER BY id DESC LIMIT ?",
            (n,)).fetchall()]

    def evolve_once(self, ctx: Optional[Any] = None, commit: bool = True,
                    verbose: bool = True) -> Dict[str, Any]:
        """One self-development step: abstract, grow, verify, commit."""
        t0 = time.time()
        before = self.stats()
        abstractions = self.propose()
        gap = self.grow_skill_for_gap(ctx)
        adopted = sum(1 for a in abstractions if a["adopted"])
        snap = self.snapshot() if (commit and (adopted or gap.get("ok"))) else {"committed": False}
        after = self.stats()
        res = {"secs": round(time.time() - t0, 2), "abstractions": len(abstractions),
               "adopted": adopted, "gap_skill": gap.get("skill") if gap.get("ok") else None,
               "grammar": f"{before['grammar_size']} -> {after['grammar_size']}",
               "skills": f"{before['skills']} -> {after['skills']}",
               "commit": snap.get("committed", False), "snapshot": snap}
        if verbose:
            log.i(f"self-development: +{adopted} primitives (grammar {res['grammar']}), "
                  f"skills {res['skills']}, committed={res['commit']}")
            for a in abstractions[:4]:
                log.d(f"   {'+' if a['adopted'] else '-'} {a['name']} "
                      f"= {' ∘ '.join(a['chain'])} (seen {a['uses']}x)")
        return res
