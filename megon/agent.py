"""MEGON I AI -- the cognitive kernel.

This is the self-deciding loop:

    perceive -> set goals -> choose an action -> act -> observe -> reflect

It chooses its own next action. The choice is not random and not scripted: it
is a UCB policy over (goal-type, tool) pairs whose values are updated from what
actually worked. Left running, it spends its attention where its own evidence
says it is weakest, writes what it finds into memory, and records why.

ABOUT THE WORD "UNRESTRICTED"
-----------------------------
MEGON decides for itself *inside an envelope*: every action passes the policy
layer, every byte counts against a budget, every execution is confined and
logged, and a pause file stops it at the next step. This is deliberate. An
agent with no envelope does not get smarter -- it gets rate-limited, IP-banned,
or it poisons its own memory, and then it has learned nothing. The envelope is
what makes unattended operation possible at all.
"""
from __future__ import annotations

import json
import math
import random
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from megon.base import *
from megon.memory import Store, SemanticIndex, Retriever
from megon.mind import Answerer, MathSolver
from megon.evals import Evaluator
from megon.learn import Tuner, learn_cycle
from megon.tools import ToolCtx, TOOLMAP, tool_catalogue, call_tool
from megon.skills import SkillLibrary
from megon.evolve import Evolution
from megon.voice import quip, banner_line

AGENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS actions(
  id INTEGER PRIMARY KEY, ts TEXT, cycle INTEGER, goal TEXT, goal_kind TEXT,
  tool TEXT, arg TEXT, ok INTEGER, reward REAL, observation TEXT, secs REAL
);
CREATE TABLE IF NOT EXISTS goals(
  id INTEGER PRIMARY KEY, ts TEXT, kind TEXT, text TEXT, priority REAL,
  status TEXT DEFAULT 'open', done_at TEXT
);
"""

# What MEGON can want. Each generator turns its own state into a concrete goal.
GOAL_KINDS = ["gap", "curiosity", "skill", "depth", "verify", "consolidate", "bootstrap"]

# The bootstrap frontier. A cold agent with no memory cannot derive goals from
# evidence it does not have yet, so it starts here and then follows its own
# findings. Once memory exists, the evidence-driven generators take over and
# this list stops being consulted.
SEED_TOPICS = [
    "transformer architecture attention mechanism",
    "retrieval augmented generation",
    "gradient descent optimization",
    "tokenization byte pair encoding",
    "reinforcement learning policy gradient",
    "knowledge graph embedding",
    "continual learning catastrophic forgetting",
    "vector database approximate nearest neighbour",
    "chain of thought reasoning",
    "mixture of experts sparse models",
    "contrastive learning representation",
    "bayesian inference posterior",
    "graph neural networks",
    "diffusion models generative",
    "program synthesis neural",
    "agent planning tool use",
    "memory augmented neural networks",
    "self supervised learning",
    "quantization low resource inference",
    "active learning query strategy",
]


class Agent:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or load_config()
        self.c = init_db()
        self.c.executescript(AGENT_SCHEMA)
        self.home = home_dir()
        self.store = Store(self.c)
        self.index = SemanticIndex(self.cfg, self.home / "index")
        self.index.load()
        self.retriever = Retriever(self.store, self.index, self.cfg)
        self.solver = MathSolver(kv_get(self.c, "solver_weights", None))
        self.answerer = Answerer(self.store, self.retriever, self.cfg, self.solver)
        self.evaluator = Evaluator(self.store, self.retriever, self.answerer, self.cfg)
        self.skills = SkillLibrary(self.c, self.home)
        self.evo = Evolution(self.cfg)          # self-development
        self.ctx = ToolCtx(self.cfg, self.c, self.store, self.retriever, self.answerer)
        self.skills.install_as_tools()
        self.pause_file = self.home / "PAUSE"
        self.log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ perceive
    def perceive(self) -> Dict[str, Any]:
        st = self.store.stats()
        last = self.c.execute(
            "SELECT name, AVG(value) v FROM metrics WHERE cycle=(SELECT MAX(cycle) FROM metrics) "
            "GROUP BY name").fetchall()
        metrics = {r["name"]: round(float(r["v"]), 4) for r in last}
        acts = self.c.execute("SELECT COUNT(*) n FROM actions").fetchone()["n"]
        return {
            "ts": now_iso(), "memory": st, "metrics": metrics,
            "actions_taken": acts, "skills": self.skills.stats(),
            "wishlist": kv_get(self.c, "wishlist", []) or [],
            "open_goals": self.c.execute(
                "SELECT COUNT(*) n FROM goals WHERE status='open'").fetchone()["n"],
            "tools": len(TOOLMAP),
            "paused": self.pause_file.exists(),
        }

    # --------------------------------------------------------------------- goals
    def set_goals(self, state: Dict[str, Any], limit: int = 6) -> List[Dict[str, Any]]:
        """Turn what MEGON knows about its own weaknesses into concrete goals."""
        made: List[Dict[str, Any]] = []

        def add(kind: str, text: str, priority: float) -> None:
            if len(made) >= limit:
                return
            if self.c.execute("SELECT 1 FROM goals WHERE text=? AND status='open'",
                              (text,)).fetchone():
                return
            self.c.execute("INSERT INTO goals(ts,kind,text,priority) VALUES(?,?,?,?)",
                           (now_iso(), kind, text, priority))
            made.append({"kind": kind, "text": text, "priority": priority})

        # 0. bootstrap: no memory means no evidence to derive goals from
        if state["memory"]["chunks"] < 250:
            done = {r["text"] for r in self.c.execute(
                "SELECT text FROM goals").fetchall()}
            open_seeds = [t for t in SEED_TOPICS
                          if f"learn about: {t}" not in done]
            for t in open_seeds[:3]:
                add("bootstrap", f"learn about: {t}", 0.9)
        m = state["metrics"]
        # 1. weakest benchmark area first -- this is the honest priority signal
        weak = sorted(((v, k) for k, v in m.items()
                       if k.endswith("_acc") or k.endswith("_f1")), key=lambda t: t[0])
        for score, name in weak[:3]:
            if score < 0.75:
                add("gap", f"improve {name} (currently {score:.2f})", 1.0 - score)
        # 2. topics it already knows it fails on
        for topic in state["wishlist"][:3]:
            add("gap", f"learn about: {topic}", 0.7)
        # 3. depth on the sources that are already proving useful
        for s in self.store.sources()[:2]:
            if s["words"] < 40_000:
                add("depth", f"deepen corpus '{s['id']}' ({human(s['words'], '')} words so far)", 0.5)
        # 4. verify what it already believes -- cheap and it catches rot
        if state["actions_taken"] > 0 and state["actions_taken"] % 7 == 0:
            add("verify", "re-verify the skill library", 0.4)
        # 5. curiosity: an under-explored corner of what it has already read
        if state["memory"]["chunks"]:
            rows = self.c.execute(
                "SELECT title FROM docs WHERE title!='' ORDER BY RANDOM() LIMIT 12").fetchall()
            titles = [r["title"] for r in rows if len(r["title"] or "") > 12]
            if titles:
                seed = random.choice(titles)
                seed = re.sub(r"\s*[-|–].*$", "", seed).strip()[:70]
                add("curiosity", f"explore adjacent territory to: {seed}", 0.35)
        # 6. consolidation keeps memory from degrading as it grows
        if state["memory"]["chunks"] > 500 and state["actions_taken"] % 11 == 0:
            add("consolidate", "rebuild index and deduplicate memory", 0.3)
        return made

    def pick_goal(self) -> Optional[sqlite3.Row]:
        return self.c.execute("SELECT * FROM goals WHERE status='open' "
                              "ORDER BY priority DESC, id LIMIT 1").fetchone()

    def close_goal(self, gid: int) -> None:
        self.c.execute("UPDATE goals SET status='done', done_at=? WHERE id=?", (now_iso(), gid))

    # -------------------------------------------------------------------- choose
    def _policy(self) -> Dict[str, Dict[str, float]]:
        return kv_get(self.c, "action_policy", {}) or {}

    def choose(self, goal: sqlite3.Row) -> Tuple[str, str]:
        """UCB over (goal-kind, tool). Genuinely learned, not a fixed script."""
        pol = self._policy()
        cands = self._candidate_tools(goal["kind"])
        if not cands:
            return "recall", goal["text"]
        total = sum(v.get("n", 0) for v in pol.values()) or 1.0
        best, best_v = None, -1e9
        for tool in cands:
            key = f"{goal['kind']}|{tool}"
            s = pol.get(key, {"n": 0, "mean": 0.0})
            v = (s["mean"] if s["n"] else 0.0) + math.sqrt(2.0 * math.log(total) / (s["n"] + 1))
            if v > best_v:
                best, best_v = tool, v
        return best, self._arg_for(goal, best)

    @staticmethod
    def _candidate_tools(kind: str) -> List[str]:
        pref = {
            "gap": ["wikipedia", "arxiv", "papers", "web_search", "hf_rows"],
            "bootstrap": ["wikipedia", "arxiv", "papers"],
            "curiosity": ["web_search", "wikipedia", "repos", "arxiv"],
            "skill": ["code_search", "repos", "run_python", "web_search"],
            "depth": ["hf_rows", "wikipedia", "fetch_page"],
            "verify": ["recall", "think", "run_python"],
            "consolidate": ["recall"],
        }
        return [t for t in pref.get(kind, ["recall"]) if t in TOOLMAP]

    @staticmethod
    def _arg_for(goal: sqlite3.Row, tool: str) -> str:
        text = goal["text"]
        m = re.search(r"(?:learn about|territory to|corpus ')([^']+)'?|: (.+)$", text)
        subject = (m.group(1) or m.group(2)).strip() if m else text
        subject = re.sub(r"\s*\(.*?\)$", "", subject).strip().lstrip(": ").strip()
        if tool == "hf_rows" and "corpus" in text:
            src = re.search(r"corpus '([^']+)'", text)
            mapping = {"fineweb-edu": "HuggingFaceFW/fineweb-edu", "fineweb": "HuggingFaceFW/fineweb",
                       "wikitext-2": "Salesforce/wikitext", "alpaca": "tatsu-lab/alpaca"}
            return mapping.get(src.group(1) if src else "", "HuggingFaceFW/fineweb-edu")
        if tool == "run_python":
            return "print(len('self-verification reachable'))"
        return subject[:200]

    # ----------------------------------------------------------------------- act
    def act(self, goal: sqlite3.Row, tool: str, arg: str) -> Dict[str, Any]:
        t0 = time.time()
        res = call_tool(tool, arg, self.ctx)
        secs = round(time.time() - t0, 2)
        obs = self._observe(tool, arg, res)
        self.c.execute(
            """INSERT INTO actions(ts,cycle,goal,goal_kind,tool,arg,ok,reward,observation,secs)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (now_iso(), int(kv_get(self.c, "cycle", 0)), goal["text"][:300], goal["kind"],
             tool, arg[:300], 1 if obs["ok"] else 0, obs["reward"],
             json.dumps(obs, ensure_ascii=False, default=str)[:4000], secs))
        self._update_policy(goal["kind"], tool, obs["reward"])
        self.log.append({"ts": now_iso(), "goal": goal["text"][:70], "tool": tool,
                         "arg": arg[:60], "reward": obs["reward"], "note": obs["note"]})
        return obs

    def _observe(self, tool: str, arg: str, res: Dict[str, Any]) -> Dict[str, Any]:
        """Turn a raw tool result into stored knowledge plus a reward signal."""
        if not isinstance(res, dict) or res.get("status", "ok").startswith(("blocked", "tool-error")):
            return {"ok": False, "reward": 0.0, "stored": 0, "note": res.get("status", "failed")}
        stored, chars = 0, 0
        if tool in ("wikipedia",):
            for p in res.get("pages", []):
                r = self.store.add_doc(p["url"], p["title"], p["text"], "agent-wikipedia", "agent")
                if r[0]:
                    stored += 1
                    chars += p["chars"]
        elif tool == "fetch_page" and res.get("text"):
            r = self.store.add_doc(arg, res.get("title", ""), res["text"], "agent-web", "agent")
            if r[0]:
                stored, chars = 1, res.get("chars", 0)
        elif tool == "arxiv":
            for p in res.get("papers", []):
                txt = f"{p['title']}\n\n{p['abstract']}"
                r = self.store.add_doc(p.get("url", ""), p["title"], txt, "agent-arxiv", "agent")
                if r[0]:
                    stored += 1
                    chars += len(txt)
        elif tool in ("web_search", "papers", "code_search", "repos"):
            blob = json.dumps(res, ensure_ascii=False, default=str)
            r = self.store.add_doc(f"agent://{tool}/{sha256(arg)[:12]}", f"{tool}: {arg[:80]}",
                                   self._flatten(tool, res), f"agent-{tool}", "agent")
            if r[0]:
                stored, chars = 1, len(blob)
        elif tool == "hf_rows":
            for row in res.get("rows", []):
                txt = str(row.get("text") or row.get("content") or "")[:200_000]
                if len(txt) > 200:
                    r = self.store.add_doc(str(row.get("url", f"hf://{arg}")), "", txt,
                                           "agent-hf", "agent")
                    if r[0]:
                        stored += 1
                        chars += len(txt)
        # reward: new, non-duplicate knowledge is what actually matters
        reward = 0.0
        if stored:
            reward = min(1.0, 0.35 + 0.05 * stored + min(0.4, chars / 200_000))
        elif tool in ("recall", "think") and res.get("hits", res.get("answer")):
            reward = 0.15
        elif res.get("ok"):
            reward = 0.2
        return {"ok": True, "reward": round(reward, 3), "stored": stored, "chars": chars,
                "note": f"{stored} new docs, {chars} chars"}

    @staticmethod
    def _flatten(tool: str, res: Dict[str, Any]) -> str:
        out = []
        for k in ("results", "works", "questions", "repos", "papers"):
            for it in res.get(k, []) or []:
                out.append(" · ".join(str(it.get(f, "")) for f in
                                      ("title", "name", "url", "link", "doi", "snippet",
                                       "desc", "abstract", "body") if it.get(f)))
        return "\n".join(x for x in out if x)[:100_000]

    def _update_policy(self, kind: str, tool: str, reward: float) -> None:
        pol = self._policy()
        s = pol.setdefault(f"{kind}|{tool}", {"n": 0.0, "mean": 0.0})
        s["n"] += 1
        s["mean"] += (reward - s["mean"]) / s["n"]
        kv_set(self.c, "action_policy", pol)

    # ------------------------------------------------------------------- reflect
    def reflect(self) -> List[str]:
        notes: List[str] = []
        cyc = int(kv_get(self.c, "cycle", 0))
        rows = self.c.execute(
            """SELECT goal_kind, tool, AVG(reward) r, COUNT(*) n FROM actions
               WHERE ts > datetime('now','-1 day') GROUP BY goal_kind, tool
               ORDER BY r DESC LIMIT 12""").fetchall()
        for r in rows:
            if r["n"] >= 2:
                note = (f"{r['goal_kind']} via {r['tool']}: mean reward {r['r']:.2f} "
                        f"over {r['n']} actions")
                self.c.execute("INSERT INTO lessons(ts,cycle,kind,text,weight) VALUES(?,?,?,?,?)",
                               (now_iso(), cyc, "agency", note, float(r["r"])))
                notes.append(note)
        # a goal that keeps failing should stop being chosen
        stuck = self.c.execute(
            """SELECT g.id, g.text, COUNT(a.id) n, AVG(a.reward) r FROM goals g
               LEFT JOIN actions a ON a.goal=g.text
               WHERE g.status='open' GROUP BY g.id HAVING n>=3 AND r<0.1""").fetchall()
        for s in stuck:
            self.c.execute("UPDATE goals SET status='abandoned' WHERE id=?", (s["id"],))
            notes.append(f"abandoned unproductive goal: {s['text'][:60]}")
        return notes

    # ---------------------------------------------------------------------- step
    def step(self, verbose: bool = True) -> Dict[str, Any]:
        if self.pause_file.exists():
            return {"paused": True}
        state = self.perceive()
        goals = self.set_goals(state)
        goal = self.pick_goal()
        if goal is None:
            log.w("no open goals -- nothing to decide this step")
            return {"paused": False, "note": "no open goals", "state": state}
        tool, arg = self.choose(goal)
        obs = self.act(goal, tool, arg)
        if obs["reward"] >= 0.35:
            self.close_goal(int(goal["id"]))
        notes = self.reflect()
        out = {"goal": goal["text"], "kind": goal["kind"], "tool": tool, "arg": arg[:80],
               "reward": obs["reward"], "note": obs["note"], "new_goals": len(goals),
               "reflection": notes, "paused": False}
        if verbose:
            log.i(f"[{goal['kind']:>11}] {tool:12s} {arg[:56]!r} -> reward {obs['reward']} "
                  f"({obs['note']})")
            # character over the commentary -- never over the numbers above
            log.d(f"      {quip('good' if obs['ok'] and obs['reward'] >= 0.35 else ('blocked' if not obs['ok'] else 'idle'))}")
            for n in notes:
                log.d(f"      reflect: {n}")
        return out

    def run(self, steps: int = 10, verbose: bool = True) -> List[Dict[str, Any]]:
        out = []
        for i in range(steps):
            if self.pause_file.exists():
                log.w("PAUSE file present -- stopping")
                break
            try:
                out.append(self.step(verbose))
            except Exception as e:
                log.e(f"step failed: {e}\n{traceback.format_exc()[-400:]}")
                out.append({"error": str(e)})
        return out

    # ------------------------------------------------------------ the value stats
    def policy_table(self) -> List[Dict[str, Any]]:
        rows = [{"key": k, "kind": k.split("|")[0], "tool": k.split("|")[1],
                 "n": int(v["n"]), "mean": round(v["mean"], 4)}
                for k, v in self._policy().items()]
        rows.sort(key=lambda r: -r["mean"])
        return rows

    def recent(self, n: int = 20) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.c.execute(
            "SELECT ts,goal_kind,tool,arg,ok,reward,observation,secs FROM actions "
            "ORDER BY id DESC LIMIT ?", (n,)).fetchall()]


# ==============================================================================
# the unattended loop
# ==============================================================================
def daemon(steps_per_wake: int = 12, sleep_s: int = 1800, cycle_every: int = 8,
           cfg: Optional[Dict[str, Any]] = None) -> None:
    """Run forever: a burst of self-chosen actions, then a full learn cycle.

    Stop it at any time with `megon agent --pause` (or touch megon_home/PAUSE).
    It checks the pause file between every single step, so it never has to be
    killed mid-action.
    """
    agent = Agent(cfg)
    log.i(banner_line())
    log.i(f"MEGON kernel online :: {len(TOOLMAP)} tools, "
          f"{agent.skills.stats()['verified']} verified skills")
    log.i(f"pause with: touch {agent.pause_file}")
    wake = 0
    while True:
        if agent.pause_file.exists():
            log.w("paused. remove the PAUSE file to resume.")
            time.sleep(30)
            continue
        wake += 1
        log.i(f"--- wake {wake}: {steps_per_wake} self-chosen actions ---")
        agent.run(steps_per_wake)
        if wake % cycle_every == 0:
            log.i("--- consolidating: full learn cycle ---")
            try:
                learn_cycle(agent.cfg)
                agent.index.build(agent.store, force=True)
            except Exception as e:
                log.w(f"learn cycle failed: {e}")
            # ---- self-development: grow its own vocabulary, commit the result ----
            try:
                ev = agent.evo.evolve_once(commit=True)
                if ev["adopted"] or ev["gap_skill"]:
                    log.i(f"  {quip('evolve')}  +{ev['adopted']} primitives "
                          f"(grammar {ev['grammar']}), gap-skill={ev['gap_skill']}")
            except Exception as e:
                log.w(f"self-development failed: {e}")
        time.sleep(sleep_s)
