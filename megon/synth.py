"""MEGON I AI -- inductive program synthesis.

This is how MEGON invents a skill it has never been given, with no model and no
GPU. You give it a name, a purpose, and a few input/output examples. It searches
a typed grammar of composable primitives for a program that satisfies every
example, then hands the survivor to the skill library, which re-verifies it in
the confined executor before it counts as acquired.

    examples: [("Hello World", "hello-world"), ("MEGON I AI", "megon-i-ai")]
    ->  slugify = join("-", map(lower, split_ws(x)))

The method is type-directed enumerative search: primitives carry signatures, so
compositions that cannot type-check are never built. That prunes the space hard
enough to search exhaustively to depth 3 on a laptop, which is the difference
between a demo and something that actually finds programs.

This is genuine inductive synthesis in the DreamCoder/FlashFill family. It will
not write a web app. It finds small, correct, reusable functions -- which is
exactly what a skill library needs, and it compounds because every synthesised
skill becomes a primitive for the next search.
"""
from __future__ import annotations

import itertools
import json
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from megon.base import *

# ==============================================================================
# the grammar: typed primitives
# ==============================================================================
S, L, N, B = "str", "list", "num", "bool"


class P:
    """A primitive with a signature, so search can type-check compositions."""

    __slots__ = ("name", "ins", "out", "fn", "code")

    def __init__(self, name: str, ins: Sequence[str], out: str,
                 fn: Callable[..., Any], code: str):
        self.name, self.ins, self.out, self.fn, self.code = name, list(ins), out, fn, code

    def __repr__(self) -> str:
        return f"{self.name}:{'+'.join(self.ins) or '_'}->{self.out}"


def _words(x: Any) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", str(x))


PRIMITIVES: List[P] = [
    # str -> str
    P("lower", [S], S, lambda x: str(x).lower(), "str(x).lower()"),
    P("upper", [S], S, lambda x: str(x).upper(), "str(x).upper()"),
    P("strip", [S], S, lambda x: str(x).strip(), "str(x).strip()"),
    P("title", [S], S, lambda x: str(x).title(), "str(x).title()"),
    P("reverse_s", [S], S, lambda x: str(x)[::-1], "str(x)[::-1]"),
    P("squash_ws", [S], S, lambda x: re.sub(r"\\s+", " ", str(x)).strip(),
      "re.sub(r'\\\\s+', ' ', str(x)).strip()"),
    P("drop_punct", [S], S, lambda x: re.sub(r"[^\\w\\s-]", "", str(x)),
      "re.sub(r'[^\\\\w\\\\s-]', '', str(x))"),
    P("digits_only", [S], S, lambda x: re.sub(r"\\D", "", str(x)), "re.sub(r'\\\\D', '', str(x))"),
    P("alpha_only", [S], S, lambda x: re.sub(r"[^A-Za-z ]", "", str(x)),
      "re.sub(r'[^A-Za-z ]', '', str(x))"),
    P("deprefix_the", [S], S,
      lambda x: re.sub(r"^the\\s+", "", str(x), flags=re.I), "re.sub(r'^the\\\\s+', '', str(x), flags=re.I)"),
    # str -> list
    P("split_ws", [S], L, lambda x: str(x).split(), "str(x).split()"),
    P("split_lines", [S], L, lambda x: str(x).splitlines(), "str(x).splitlines()"),
    P("split_comma", [S], L, lambda x: [p.strip() for p in str(x).split(",")],
      "[p.strip() for p in str(x).split(',')]"),
    P("chars", [S], L, lambda x: list(str(x)), "list(str(x))"),
    P("words_re", [S], L, _words, "re.findall(r'[A-Za-z0-9]+', str(x))"),
    # list -> list
    P("dedupe", [L], L, lambda x: list(dict.fromkeys(x)), "list(dict.fromkeys(x))"),
    P("sort_l", [L], L, lambda x: sorted(x, key=str), "sorted(x, key=str)"),
    P("reverse_l", [L], L, lambda x: list(reversed(x)), "list(reversed(x))"),
    P("drop_empty", [L], L, lambda x: [i for i in x if str(i).strip()],
      "[i for i in x if str(i).strip()]"),
    # list -> str
    P("join_ws", [L], S, lambda x: " ".join(str(i) for i in x), "' '.join(str(i) for i in x)"),
    P("join_nl", [L], S, lambda x: chr(10).join(str(i) for i in x),
      "chr(10).join(str(i) for i in x)"),
    P("join_dash", [L], S, lambda x: "-".join(str(i) for i in x), "'-'.join(str(i) for i in x)"),
    P("join_under", [L], S, lambda x: "_".join(str(i) for i in x), "'_'.join(str(i) for i in x)"),
    P("join_empty", [L], S, lambda x: "".join(str(i) for i in x), "''.join(str(i) for i in x)"),
    P("first", [L], S, lambda x: str(x[0]) if x else "", "str(x[0]) if x else ''"),
    P("last", [L], S, lambda x: str(x[-1]) if x else "", "str(x[-1]) if x else ''"),
    P("concat_l", [L], S, lambda x: " ".join(str(i) for i in x), "' '.join(str(i) for i in x)"),
    # str -> num
    P("len_s", [S], N, lambda x: len(str(x)), "len(str(x))"),
    P("count_words", [S], N, lambda x: len(_words(x)), "len(re.findall(r'[A-Za-z0-9]+', str(x)))"),
    P("to_num", [S], N, lambda x: float(re.sub(r"[^\\d.\\-]", "", str(x)) or 0),
      "float(re.sub(r'[^\\\\d.\\\\-]', '', str(x)) or 0)"),
    # num -> num
    P("abs_n", [N], N, lambda x: abs(float(x)), "abs(float(x))"),
    P("round_n", [N], N, lambda x: round(float(x)), "round(float(x))"),
    P("neg", [N], N, lambda x: -float(x), "-float(x)"),
    # list -> num
    P("len_l", [L], N, lambda x: len(x), "len(x)"),
    P("sum_l", [L], N, lambda x: sum(float(i) for i in x), "sum(float(i) for i in x)"),
    # num -> str
    P("num_to_s", [N], S, lambda x: str(int(x)) if float(x).is_integer() else str(x),
      "str(int(x)) if float(x).is_integer() else str(x)"),
]

BY_OUT: Dict[str, List[P]] = {}
for _p in PRIMITIVES:
    BY_OUT.setdefault(_p.out, []).append(_p)

# map(f, xs): apply a str->str primitive to every element of a list
MAP_P = P("map", [S, L], L, None, "")


# ==============================================================================
# programs
# ==============================================================================
class Prog:
    """A composition: apply .chain[-1] last. Rendered to real Python source."""

    __slots__ = ("chain", "out")

    def __init__(self, chain: List[P]):
        self.chain = chain
        self.out = chain[-1].out if chain else S

    def __call__(self, x: Any) -> Any:
        v = x
        for p in self.chain:
            v = p.fn(v)
        return v

    def expr(self) -> str:
        e = "x"
        for p in self.chain:
            e = p.code.replace("str(x)", f"str({e})").replace("(x)", f"({e})") \
                if p.name not in ("len_s", "len_l", "sum_l") else p.code.replace("(x)", f"({e})")
            if p.name in ("len_s", "len_l", "count_words", "to_num", "sum_l"):
                e = p.code.replace("(x)", f"({e})")
        return e

    def source(self) -> str:
        """Standalone, importable Python -- no reference to this module."""
        lines = ["import re", "", "", "def run(x):"]
        e = "x"
        for p in self.chain:
            e = _render(p, e)
        lines.append(f"    return {e}")
        return "\n".join(lines) + "\n"

    def label(self) -> str:
        return " ∘ ".join(reversed([p.name for p in self.chain])) or "identity"


def _render(p: P, inner: str) -> str:
    """Substitute the inner expression into a primitive's code template."""
    code = p.code
    if p.name in ("len_s", "len_l", "sum_l", "count_words", "to_num"):
        return code.replace("(x)", f"({inner})")
    # primitives are written with a literal x; wrap it
    return code.replace("x", inner) if _safe_sub(code) else f"({code})"


def _safe_sub(code: str) -> bool:
    """Only substitute when 'x' appears as a standalone identifier."""
    return re.search(r"(?<![\w.])x(?![\w])", code) is not None


# ==============================================================================
# the search
# ==============================================================================
class Synthesiser:
    """Type-directed enumerative search over compositions of primitives."""

    def __init__(self, max_depth: int = 3, max_nodes: int = 120_000, time_s: float = 25.0):
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.time_s = time_s

    def _candidates(self, out_type: str, depth: int,
                    deadline: float) -> List[Prog]:
        """All well-typed programs of length <= depth producing out_type."""
        if depth <= 0:
            return []
        found: List[Prog] = []
        # depth 1: a single primitive
        for p in BY_OUT.get(out_type, []):
            if len(p.ins) == 1:
                found.append(Prog([p]))
            if time.time() > deadline:
                return found
        if depth == 1:
            return found
        # depth n: a primitive whose output is out_type, fed by a program
        # producing that primitive's input type
        for p in BY_OUT.get(out_type, []):
            if len(p.ins) != 1:
                continue
            inner_t = p.ins[0]
            for sub in self._candidates(inner_t, depth - 1, deadline):
                found.append(Prog(sub.chain + [p]))
                if len(found) > self.max_nodes or time.time() > deadline:
                    return found
        return found

    @staticmethod
    def _matches(got: Any, want: Any) -> bool:
        if isinstance(want, float) and isinstance(got, (int, float)):
            return abs(float(got) - want) < 1e-6
        if isinstance(want, (int, float)) and isinstance(got, (int, float)):
            return float(got) == float(want)
        return got == want

    def _score(self, prog: Prog, examples: List[Tuple[Any, Any]]) -> Tuple[int, bool]:
        """Returns (passed, all_passed). Exceptions score as failures, not crashes."""
        n = 0
        for inp, want in examples:
            try:
                if self._matches(prog(inp), want):
                    n += 1
                else:
                    return n, False
            except Exception:
                return n, False
        return n, True

    def synthesise(self, examples: List[Tuple[Any, Any]], out_type: str = S,
                   held_out: Optional[List[Tuple[Any, Any]]] = None,
                   auto_split: bool = True, prefer_short: bool = True) -> Dict[str, Any]:
        """Search for the simplest program consistent with the examples.

        A program that merely fits the examples you gave is not a skill -- it is
        a coincidence. `join_empty ∘ dedupe` passes [("a\nb\na","a\nb")] by
        deduplicating *characters* and then happens to reassemble them. So a
        candidate must also survive examples the search never saw; if it fails
        them, the search rejects it and keeps looking instead of returning a
        plausible-looking overfit.

        `auto_split` holds out the last example whenever there are at least
        three, which is what makes this work when you only supply a few.
        """
        if not examples:
            return {"ok": False, "reason": "no examples"}
        held_out = list(held_out or [])
        train = list(examples)
        if auto_split and not held_out and len(train) >= 3:
            held_out = train[-1:]
            train = train[:-1]
        deadline = time.time() + self.time_s
        t0 = time.time()
        tried = rejected = 0
        best: Optional[Prog] = None
        for depth in range(1, self.max_depth + 1):
            for prog in self._candidates(out_type, depth, deadline):
                tried += 1
                if tried > self.max_nodes or time.time() > deadline:
                    break
                _, fit = self._score(prog, train)
                if not fit:
                    continue
                if held_out:
                    _, gen = self._score(prog, held_out)
                    if not gen:
                        rejected += 1          # overfit: keep searching
                        continue
                if best is None or (prefer_short and len(prog.chain) < len(best.chain)):
                    best = prog
                break                          # depth-ordered => shortest first
            if best is not None or time.time() > deadline:
                break
        secs = round(time.time() - t0, 2)
        if best is None:
            return {"ok": False, "reason": "no program generalised within budget",
                    "tried": tried, "rejected_overfit": rejected, "secs": secs}
        return {"ok": True, "program": best.label(), "source": best.source(),
                "tests": [[i, w] for i, w in list(train) + held_out],
                "tried": tried, "rejected_overfit": rejected,
                "depth": len(best.chain), "secs": secs,
                "held_out": len(held_out)}


# ==============================================================================
# a small demo set, so the mechanism is testable on its own
# ==============================================================================
DEMO_TASKS: Dict[str, Dict[str, Any]] = {
    "slugify": {
        "purpose": "turn a title into a url slug",
        "examples": [("Hello World", "hello-world"), ("MEGON I AI", "megon-i-ai"),
                     ("Deep  Learning", "deep-learning")],
        "out": S,
    },
    "snake_case": {
        "purpose": "turn a title into a snake_case identifier",
        "examples": [("Hello World", "hello_world"), ("MEGON I AI", "megon_i_ai")],
        "out": S,
    },
    "shout": {
        "purpose": "uppercase and strip surrounding space",
        "examples": [("  hi  ", "HI"), ("megon", "MEGON")],
        "out": S,
    },
    "word_count": {
        "purpose": "count the alphanumeric words in a string",
        "examples": [("one two three", 3), ("hello", 1)],
        "out": N,
    },
    "lower_words": {
        "purpose": "lowercase a phrase, keeping single spaces",
        "examples": [("Hello   World", "hello world"), ("MEGON I AI", "megon i ai"),
                     ("Deep  Learning", "deep learning")],
        "out": S,
    },
    "shout_words": {
        "purpose": "uppercase a phrase, keeping single spaces",
        "examples": [("hello   world", "HELLO WORLD"), ("megon i ai", "MEGON I AI"),
                     ("deep  learning", "DEEP LEARNING")],
        "out": S,
    },
    "title_words": {
        "purpose": "title-case a phrase, keeping single spaces",
        "examples": [("hello   world", "Hello World"), ("megon i ai", "Megon I Ai"),
                     ("deep  learning", "Deep Learning")],
        "out": S,
    },
    "unique_lines": {
        "purpose": "deduplicate lines and rejoin",
        "examples": [("a\nb\na", "a\nb"), ("x\nx\ny", "x\ny"),
                     ("p\nq\np\nr", "p\nq\nr")],
        "out": S,
    },
}


def synthesise_demo(name: str) -> Dict[str, Any]:
    task = DEMO_TASKS[name]
    return Synthesiser().synthesise(task["examples"], task["out"])
