"""MEGON I AI -- learning from real code.

MEGON could already *find* repositories. It could not read them. This closes
that gap: it pulls real Python source from public GitHub repositories, extracts
the pure functions -- the ones with no I/O, no globals, no side effects -- and
tries to adopt them as skills.

Three things make this safe, and they are the whole design:

1. **Read widely, copy narrowly.** Every public repository may be read --
   reading public code is research. The licence decides only what may be
   *copied*: permissive (MIT, BSD, Apache-2.0, ISC, Unlicense, CC0) means the
   code itself is adopted, anything else means MEGON keeps the interface and
   intent and writes its own implementation. Attribution travels with both.
2. **Static screening.** A function is a candidate only if its AST contains no
   imports, no open/exec/eval/compile, no network, no dunder tricks, no global
   mutation. Reading is not enough to trust code.
3. **Execution proves it.** Every candidate runs in the confined interpreter
   with generated probes before it counts. Code that does not survive contact
   with a test is not a skill, however popular its repository.

MEGON reads to learn patterns, the way a developer reads a library. It does not
vendor code into itself wholesale, and it keeps the licence with anything it
keeps.
"""
from __future__ import annotations

import ast
import json
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

from megon.base import *

# ---------------------------------------------------------------- licences
PERMISSIVE = {
    "mit", "bsd-2-clause", "bsd-3-clause", "apache-2.0", "isc",
    "unlicense", "cc0-1.0", "0bsd", "python-2.0",
}
# Ambiguity resolves to "do not copy", never to "probably fine". Reading is
# always allowed; it is copying that needs a positive licence.
LICENSE_OK = re.compile(
    r"\b(mit|bsd|apache[- ]?2|isc|unlicense|cc0|public domain)\b", re.I)

RAW = "https://raw.githubusercontent.com"
API_REPO = "https://api.github.com/repos"
API_SEARCH = "https://api.github.com/search"

# ---------------------------------------------------------------- screening
BANNED_CALLS = {
    "open", "exec", "eval", "compile", "input", "__import__", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
    "exit", "quit", "system", "popen", "spawn", "fork",
}
BANNED_ATTRS = {"system", "popen", "urlopen", "requests", "socket", "path",
                "environ", "subprocess", "shutil", "remove", "unlink", "write"}
MAX_SRC = 4000          # a learned function stays small or it is not a pattern
MAX_ARGS = 3


def licence_of(meta: Dict[str, Any]) -> Tuple[bool, str]:
    """True only if we can positively identify a permissive licence."""
    lic = (meta.get("license") or {})
    key = (lic.get("spdx_id") or "").lower()
    name = (lic.get("name") or "")
    if key and key != "noassertion" and key in PERMISSIVE:
        return True, key
    if key == "noassertion":
        return False, "unspecified (skipped)"
    if LICENSE_OK.search(name):
        return True, name[:40]
    return False, (name or "none declared")[:40]


def screen(src: str) -> Tuple[bool, str]:
    """Static safety screen. Returns (ok, reason)."""
    if len(src) > MAX_SRC:
        return False, "too large"
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"syntax:{e.msg}"
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return False, "not a single function"
    fn = tree.body[0]
    if len(fn.args.args) == 0 or len(fn.args.args) > MAX_ARGS:
        return False, "arity"
    if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs:
        return False, "complex signature"
    if any(d is not None for d in fn.args.defaults):
        return False, "default args"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "imports"
        if isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
            return False, "mutates scope"
        if isinstance(node, ast.Name) and isinstance(node.id, str):
            if node.id in BANNED_CALLS:
                return False, f"banned:{node.id}"
            if node.id.startswith("__"):
                return False, "dunder"
        if isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            return False, f"banned attr:{node.attr}"
        if isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.Yield,
                             ast.YieldFrom)):
            return False, "async/generator"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) > 2000:
                return False, "embedded blob"
    if not ast.get_docstring(fn):
        return False, "no docstring"
    return True, "clean"


# ---------------------------------------------------------------- discovery
def find_repos(query: str, n: int, ctx) -> List[Dict[str, Any]]:
    """Find well-licensed, reasonably popular repos for a topic.

    GitHub's repo search matches the whole phrase against name, description and
    readme, so a multi-word query like "string utilities" returns almost
    nothing. Searching the words in the description field separately, and
    falling back to a single distinctive term, is what actually finds repos.
    """
    terms = [t for t in re.split(r"\s+", query.strip()) if len(t) > 2]
    attempts = [
        f"{query} language:python stars:>200",
        " ".join(f"in:description,readme {t}" for t in terms[:2]) + " language:python stars:>200",
    ]
    if terms:
        attempts.append(f"{max(terms, key=len)} language:python stars:>500")

    seen, out = set(), []
    for qtext in attempts:
        if len(out) >= n:
            break
        q = urllib.parse.urlencode({"q": qtext, "per_page": n * 6, "sort": "stars"})
        data, _status = ctx.get(f"{API_SEARCH}/repositories?{q}", max_bytes=400_000)
        if data is None:
            continue
        try:
            items = json.loads(data.decode("utf-8", "replace")).get("items", [])
        except Exception:
            continue
        for it in items:
            name = it.get("full_name")
            if not name or name in seen:
                continue
            ok, why = licence_of(it)
            # No licence gate on *reading*. Public code is public; reading it is
            # research. What the licence governs is whether we may copy it, and
            # that is decided later, in _try_adopt.
            seen.add(name)
            out.append({"full_name": name, "license": why, "permissive": ok,
                        "stars": it.get("stargazers_count"),
                        "desc": (it.get("description") or "")[:160],
                        "default_branch": it.get("default_branch") or "main",
                        "url": it.get("html_url")})
            if len(out) >= n:
                break
    return out


def list_py_files(repo: str, branch: str, ctx, limit: int = 30,
                  depth: int = 2) -> List[str]:
    """List .py files in a repo, walking into package directories.

    Real code almost never sits at the repository root -- it lives in the
    package folder. A root-only listing finds `setup.py` and nothing else,
    which is why the first version of this learned zero functions.
    """
    found: List[str] = []

    def walk(dirpath: str, level: int) -> None:
        if len(found) >= limit or level > depth:
            return
        ref = f"{API_REPO}/{repo}/contents"
        if dirpath:
            ref += f"/{urllib.parse.quote(dirpath)}"
        data, _ = ctx.get(f"{ref}?ref={branch}", max_bytes=200_000)
        if data is None:
            return
        try:
            rows = json.loads(data.decode("utf-8", "replace"))
        except Exception:
            return
        if not isinstance(rows, list):
            return
        dirs = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            nm = r.get("name", "")
            if r.get("type") == "file" and nm.endswith(".py"):
                if nm.startswith(("test", "conftest", "setup")):
                    continue
                if nm in {"__init__.py", "__main__.py"}:
                    continue   # re-exports, not implementations
                found.append(r["path"])
                if len(found) >= limit:
                    return
            elif r.get("type") == "dir":
                # skip the directories that never hold the library's own code
                if nm.startswith((".", "_")) or nm in {
                        "tests", "test", "docs", "examples", "benchmarks",
                        "scripts", ".github"}:
                    continue
                dirs.append(r["path"])
        for d in dirs:
            if len(found) >= limit:
                return
            walk(d, level + 1)

    walk("", 1)
    return found[:limit]


def read_source(repo: str, branch: str, path: str, ctx) -> Optional[str]:
    url = f"{RAW}/{repo}/{branch}/{urllib.parse.quote(path)}"
    data, status = ctx.get(url, max_bytes=200_000)
    if data is None:
        return None
    return data.decode("utf-8", "replace")


def extract_functions(src: str) -> List[Dict[str, Any]]:
    """Pull single, self-contained, documented functions out of a module.

    Handles both module-level functions and `@staticmethod`s. Most real
    libraries are class-based, so a module-level-only extractor finds nothing
    in the majority of repositories.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []

    def consider(node: ast.AST) -> None:
        if not isinstance(node, ast.FunctionDef):
            return
        if node.name.startswith("_"):
            return
        # for methods, only @staticmethod is self-contained enough to lift out
        deco = [ast.unparse(d) for d in node.decorator_list]
        is_static = any(d in ("staticmethod", "builtins.staticmethod") for d in deco)
        if node.decorator_list and not is_static:
            return
        seg = ast.get_source_segment(src, node)
        if not seg:
            return
        if is_static:
            seg = seg.replace("@staticmethod\n", "", 1)
        ok, _why = screen(seg)
        if not ok:
            return
        out.append({
            "name": node.name,
            "source": seg,
            "doc": (ast.get_docstring(node) or "").strip()[:300],
            "args": [a.arg for a in node.args.args],
        })

    for node in tree.body:
        consider(node)
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                consider(sub)
    return out


# ---------------------------------------------------------------- learning
def learn_from_code(query: str, ctx, skills=None, max_repos: int = 3,
                    verbose: bool = True) -> Dict[str, Any]:
    """Read real open-source code and adopt what survives testing.

    Returns counts plus the attribution record for anything adopted.
    """
    repos = find_repos(query, max_repos, ctx)
    if verbose:
        log.i(f"code learning: {len(repos)} public repos for {query!r}")
    adopted, screened, probed, attributions = 0, 0, 0, []
    for r in repos:
        files = list_py_files(r["full_name"], r["default_branch"], ctx)
        if verbose:
            log.d(f"  {r['full_name']} [{r['license']}] -- {len(files)} candidate files")
        for path in files:
            src = read_source(r["full_name"], r["default_branch"], path, ctx)
            if not src:
                continue
            for fn in extract_functions(src):
                screened += 1
                res = _try_adopt(fn, r, path, ctx, skills)
                probed += 1
                if res.get("adopted"):
                    adopted += 1
                    attributions.append(res["attribution"])
                    if verbose:
                        log.i(f"    + adopted {fn['name']} from "
                              f"{r['full_name']}/{path} [{r['license']}]")
    return {"repos": len(repos), "screened": screened, "probed": probed,
            "adopted": adopted, "attributions": attributions, "status": "ok"}


def _try_adopt(fn: Dict[str, Any], repo: Dict[str, Any], path: str,
               ctx, skills) -> Dict[str, Any]:
    """Run the function on probes, pin what it did, then adopt it.

    We do not know the correct outputs of someone else's function, so we cannot
    assert them. What we can do is *characterise*: run it once in the confined
    executor, record what it returned, and make that the test suite. From then on
    the library re-runs those exact cases and the skill stops being verified the
    moment behaviour drifts.

    Be honest about what that buys: it proves the function is total on ordinary
    inputs and stable, not that it is correct. Correctness was the upstream
    author's job and their licence's promise.
    """
    if skills is None:
        return {"adopted": False, "reason": "no skill library"}

    # ---- the licence decides *what* we may keep, not whether we may look ----
    if not repo.get("permissive"):
        return _learn_idea(fn, repo, path, ctx)

    probes = _probe_values(len(fn["args"]))
    if not probes:
        return {"adopted": False, "reason": "no probes"}

    harness = (
        "import json\n" + fn["source"] + "\n"
        "_probes = " + repr(probes) + "\n"
        "_out = []\n"
        "for _a in _probes:\n"
        "    try:\n"
        "        _r = " + fn["name"] + "(*_a)\n"
        "        _j = json.dumps(_r)\n"          # must be serialisable to be pinnable
        "    except Exception:\n"
        "        continue\n"
        "    else:\n"
        "        _out.append(list(_a) + [json.loads(_j)])\n"
        "print(json.dumps(_out))\n"
    )
    from megon.tools import run_python
    res = run_python(harness, 20, ctx)
    if not res.get("ok"):
        return {"adopted": False, "reason": res.get("status", "failed")}
    try:
        tests = json.loads((res.get("stdout") or "").strip().splitlines()[-1])
    except Exception as e:
        return {"adopted": False, "reason": f"unpinnable:{e}"}
    if len(tests) < 2:
        return {"adopted": False, "reason": f"only {len(tests)} probes survived"}

    attribution = {
        "function": fn["name"],
        "repo": repo["full_name"], "path": path,
        "license": repo["license"], "url": repo.get("url"),
        "doc": fn["doc"],
    }
    header = (f"# learned from {repo['full_name']} ({repo['license']})\n"
              f"# {repo.get('url') or ''}/{path}\n"
              f"# MEGON I AI, created by Megix -- owner Redoyanul Haque\n")
    # The library's verifier calls `run(*args)`. A function lifted out of
    # someone else's module keeps its own name, so without this alias every
    # characterisation test fails on NameError -- which is exactly what the
    # first run reported: 8 tests, 0 passed.
    body = header + fn["source"] + f"\n\nrun = {fn['name']}\n"
    try:
        skills.add(fn["name"], body, tests,
                   fn["doc"] or "learned from open source",
                   "learned-from-code",
                   origin=f"{repo['full_name']} [{repo['license']}]",
                   notes="characterisation tests; licence recorded")
    except Exception as e:
        return {"adopted": False, "reason": f"add:{e}"}
    v = skills.verify(fn["name"], ctx)
    if not v.get("ok"):
        return {"adopted": False, "reason": f"verify:{v.get('detail')}"}
    return {"adopted": True, "attribution": attribution, "tests": len(tests)}


def _probe_values(n: int) -> List[List[Any]]:
    """Ordinary inputs, weighted toward the types real utilities accept.

    The first version of this probed mostly integers. That is why every real
    function failed: a string helper raises on `f(1)`, so nothing survived and
    nothing could be characterised. Strings, lists and dicts are what
    utility libraries actually take, so they come first.
    """
    one: List[Any] = [
        "hello world", "Hello World", "abc", "", "some-text_here",
        "  spaced  ", "MiXeD CaSe", "a,b,c",
        [1, 2, 3], ["a", "b"], [], {},
        {"a": 1, "b": 2}, {"key": "value"},
        1, 0, -1, 3.5, 10,
        True,
    ]
    if n == 1:
        return [[v] for v in one]
    two: List[List[Any]] = [
        ["hello world", "-"], ["a b c", "_"], ["Hello", "world"],
        ["some text", ","], [[1, 2], [3, 4]], [[1, 2], 0],
        [{"a": 1}, "a"], [{"a": 1}, "b"], ["abc", 2], ["abc", 0],
        [3, 1], [1, 2], [10, 3], [[1, 2, 3], 1],
    ]
    if n == 2:
        return two
    return [["a b c", "-", 2], [[1, 2, 3], 0, 1], [{"a": 1}, "a", "b"],
            ["hello world", "_", 0], [1, 2, 3]]


def _learn_idea(fn: Dict[str, Any], repo: Dict[str, Any], path: str,
                ctx) -> Dict[str, Any]:
    """Learn what a function does, without copying its code.

    For a repository whose licence is absent or restrictive, MEGON may still
    read it -- reading public code is research -- but it must not copy the
    implementation into a project the user publishes. Copyright protects the
    expression, not the idea, so what is kept is the *interface and intent*:
    the name, the arguments, the docstring, the shape of the problem.

    MEGON then writes its own implementation from that description via the
    synthesiser. Same capability acquired, no copied expression, and nothing
    for the user to unwind later if they publish this.
    """
    idea = {
        "name": fn["name"],
        "args": fn["args"],
        "doc": fn["doc"],
        "repo": repo["full_name"],
        "license": repo["license"],
        "path": path,
        "kind": "learned-idea",
        "note": "interface and intent only; implementation not copied",
    }
    # The text has to be substantive: the memory store rejects short strings,
    # so a bare "pattern f(x)" line was silently dropped as too-short.
    args = ", ".join(fn["args"]) or "no arguments"
    body = (
        f"Learned interface (implementation deliberately not copied).\n"
        f"Function: {fn['name']}({args})\n"
        f"Purpose: {fn['doc'] or 'no docstring supplied'}\n"
        f"Observed in: {repo['full_name']}/{path} under licence '{repo['license']}'.\n"
        f"MEGON should reimplement this behaviour from the description above using "
        f"its own synthesiser rather than copying the source, because the licence "
        f"does not permit redistribution of the expression. The idea itself -- the "
        f"signature and the intent -- is what is being retained here."
    )
    from megon.tools import remember
    res = remember(body, title=f"learned idea: {fn['name']}",
                   url=repo.get("url") or "",
                   source_id=f"codex:{repo['full_name']}", ctx=ctx)
    idea["stored"] = bool(res.get("ok"))
    return {"adopted": False, "idea": True, "reason": "idea-only (licence)",
            "attribution": idea}
