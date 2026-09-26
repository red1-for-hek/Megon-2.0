"""MEGON I AI -- command line interface"""
from __future__ import annotations
from megon.base import *
from megon.memory import *
from megon.mind import *
from megon.evals import *
from megon.learn import *
from megon.tools import ToolCtx, TOOLMAP, call_tool

# ==============================================================================
# 18. CLI
# ==============================================================================
BANNER = r"""
  __  __ _____ ____  ___  _   _      _    ___
 |  \/  | ____/ ___|/ _ \| \ | |    / \  |_ _|
 | |\/| |  _|| |  _| | | |  \| |   / _ \  | |
 | |  | | |__| |_| | |_| | |\  |  / ___ \ | |
 |_|  |_|_____\____|\___/|_| \_| /_/   \_\___|
        self-deciding cognitive engine  v{v}
        created by Megix · owner Redoyanul Haque
"""


def cmd_status(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    c = init_db()
    store = Store(c)
    st = store.stats()
    index = SemanticIndex(cfg, home_dir() / "index")
    loaded = index.load()
    lb = store.leaderboard()
    print(BANNER.format(v=VERSION))
    print(f"  home       {home_dir()}")
    print(f"  database   {db_path()}  ({human(db_path().stat().st_size if db_path().exists() else 0)})")
    print(f"  corpus     {st['docs']} docs · {st['chunks']} chunks · {human(st['bytes'])} on disk · "
          f"~{human(st['tokens_approx'], '')} tokens")
    print(f"  index      {'loaded' if loaded else 'not built'} · {len(index.ids)} vectors · "
          f"backend {'lsa+bm25' if SKLEARN else 'hash+bm25'} · dim {index.dim}")
    print(f"  benchmark  {st['eval_items']} items · {len(lb)} measured cycles")
    print(f"  budget     {cfg['budget']['mb_per_day']} MB/day · {cfg['budget']['max_docs_per_day']} docs/day · "
          f"≤{cfg['budget']['max_requests_per_min']} req/min")
    print(f"  deps       numpy={'yes' if np is not None else 'NO'} sklearn={'yes' if SKLEARN else 'NO'} "
          f"bs4={'yes' if BeautifulSoup is not None else 'NO'} requests={'yes' if requests is not None else 'NO'} "
          f"llm={'yes' if llm_ready(cfg) else 'no (extractive mode)'}")
    if lb:
        last = lb[-1]
        print(f"  last cycle #{last['cycle']} @ {last['ts']} :: score {last.get('score', 0)}")
    else:
        print("  last cycle  none yet -- run: python3 megon learn")
    print()
    return 0


def cmd_learn(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    rep = learn_cycle(cfg, mb=args.mb, docs=args.docs, minutes=args.minutes,
                      verbose=args.verbose, skip_tune=args.no_tune)
    ph = rep.get("phases", {})
    print()
    print(f"cycle {rep['cycle']} complete :: score "
          f"{ph.get('tune', {}).get('final', ph.get('evaluate', {}).get('baseline', {})).get('score', 0)}")
    print(f"version card: {ph.get('publish', {}).get('card')}")
    return 0


def cmd_ask(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    c = init_db()
    store = Store(c)
    index = SemanticIndex(cfg, home_dir() / "index")
    index.load()
    retriever = Retriever(store, index, cfg)
    ov = kv_get(c, "best_overrides", {}) or {}
    answerer = Answerer(store, retriever, cfg, MathSolver(kv_get(c, "solver_weights", None)))
    q = " ".join(args.question)
    if not q:
        print("ask: nothing to ask"); return 2
    a = answerer.answer(q, overrides=ov)
    print()
    print(textwrap.fill(a["answer"], 100, initial_indent="  ", subsequent_indent="  "))
    print()
    if a["citations"]:
        print("  evidence:")
        for ci in a["citations"][:6]:
            print(f"    [{ci['n']}] {(ci['title'] or ci['url'])[:78]}  (score {ci['score']})")
            print(f"        {ci['url'][:96]}")
    if a.get("math") and a["math"].get("answer") is not None:
        print(f"  arithmetic: {a['math']['expr']} = {int(a['math']['answer'])} "
              f"[strategy {a['math']['strategy']}, conf {a['math']['confidence']}]")
    print(f"  method {a['method']} · confidence {a['confidence']} · {a['ms']} ms · "
          f"{a['corpus']['chunks']} chunks searched")
    print()
    return 0


def cmd_chat(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    c = init_db()
    store = Store(c)
    index = SemanticIndex(cfg, home_dir() / "index")
    index.load()
    retriever = Retriever(store, index, cfg)
    ov = kv_get(c, "best_overrides", {}) or {}
    answerer = Answerer(store, retriever, cfg, MathSolver(kv_get(c, "solver_weights", None)))
    print(BANNER.format(v=VERSION))
    print("  type a question · 'memory <q>' to search raw memory · 'learn' for a cycle · Ctrl-D to quit\n")
    while True:
        try:
            line = input("mythos> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("quit", "exit"):
            break
        if line == "learn":
            learn_cycle(cfg, mb=args.mb or 2.0, docs=args.docs or 10, minutes=args.minutes or 5)
            index.build(store, force=True)
            continue
        if line.startswith("memory "):
            for h in retriever.search(line[7:], k=5, overrides=ov):
                print(f"  [{h['score']:.3f}] {(h['title'] or h['url'])[:70]}")
                print(textwrap.fill(h["text"][:300], 96, initial_indent="        ", subsequent_indent="        "))
            continue
        a = answerer.answer(line, overrides=ov)
        print()
        print(textwrap.fill(a["answer"], 96, initial_indent="  ", subsequent_indent="  "))
        for ci in a["citations"][:4]:
            print(f"    [{ci['n']}] {(ci['title'] or ci['url'])[:80]}")
        print()
    return 0


def cmd_eval(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    c = init_db()
    store = Store(c)
    index = SemanticIndex(cfg, home_dir() / "index")
    index.load()
    retriever = Retriever(store, index, cfg)
    answerer = Answerer(store, retriever, cfg, MathSolver(kv_get(c, "solver_weights", None)))
    ev = Evaluator(store, retriever, answerer, cfg)
    if args.generate:
        print("generated:", ev.generate_items(args.generate))
    r = ev.run(limit=args.limit, verbose=args.verbose)
    print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
    return 0


def cmd_ingest(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    c = init_db()
    store = Store(c)
    budget = Budget(args.mb or 50, args.docs or 200, args.minutes or 30, cfg)
    rate = RateLimiter(cfg["budget"]["min_request_interval_s"], cfg["budget"]["max_requests_per_min"])
    policy = Policy(cfg)
    for target in args.targets:
        if target.startswith(("http://", "https://")):
            src = {"id": urllib.parse.urlparse(target).netloc, "kind": "url", "url": target}
        else:
            p = Path(target).expanduser()
            if not p.exists():
                log.e(f"not found: {target}"); continue
            txt = p.read_text(encoding="utf-8", errors="replace")
            if not policy.check_content(txt, str(p), c):
                continue
            doc_id, why = store.add_doc(f"file://{p}", p.name, txt, "local-file", "file")
            print(f"  {p.name}: {doc_id or why}")
            continue
        r = ingest_source(src, cfg, store, budget, rate, policy)
        r["skipped"] = dict(r["skipped"]) if isinstance(r["skipped"], collections.Counter) else r["skipped"]
        print(f"  {json.dumps(r, ensure_ascii=False, default=str)[:400]}")
    index = SemanticIndex(cfg, home_dir() / "index")
    print("rebuilding index:", json.dumps(index.build(store, force=True)))
    return 0


def cmd_leaderboard(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    c = init_db()
    lb = Store(c).leaderboard()
    if not lb:
        print("no measured cycles yet -- run: python3 megon learn")
        return 0
    hdr = f"{'cycle':>6} {'score':>8} {'cloze':>7} {'qa_f1':>7} {'retr':>6} {'math':>6} {'docs':>7}  when"
    print(hdr); print("-" * len(hdr))
    for r in lb:
        print(f"{r['cycle']:>6} {r.get('score',0):>8.4f} {r.get('cloze_acc',0):>7.3f} "
              f"{r.get('qa_f1',0):>7.3f} {r.get('retrieval_r1',0):>6.3f} {r.get('math_acc',0):>6.3f} "
              f"{int(r.get('corpus_docs',0)):>7}  {r['ts']}")
    if len(lb) >= 2:
        d = lb[-1].get("score", 0) - lb[0].get("score", 0)
        print("-" * len(hdr))
        print(f"improvement over {len(lb)} cycles: {'+' if d >= 0 else ''}{d:.4f}")
    return 0


def cmd_sources(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    c = init_db()
    have = {s["id"]: s for s in Store(c).sources()}
    print(f"{'id':<22}{'priority':>9}{'docs':>7}{'words':>10}  note")
    print("-" * 100)
    for s in cfg.get("sources", []):
        h = have.get(s["id"], {})
        note = s.get("note") or s.get("url") or s.get("dataset", "")
        print(f"{s['id']:<22}{s.get('priority',0.5):>9}{h.get('docs',0):>7}"
              f"{human(h.get('words',0),'w'):>10}  {str(note)[:60]}")
    extra = [k for k in have if k not in {s['id'] for s in cfg.get('sources', [])}]
    for k in extra:
        print(f"{k:<22}{'-':>9}{have[k]['docs']:>7}{human(have[k]['words'],'w'):>10}  (ad-hoc ingest)")
    return 0


def cmd_wishlist(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    c = init_db()
    wl = kv_get(c, "wishlist", []) or []
    print(f"{len(wl)} topics queued for the next targeted crawl (fed by evaluator failures):")
    for t in wl:
        print("  -", t)
    if args.add:
        wl.extend(args.add)
        kv_set(c, "wishlist", wl[-40:])
        print("added.")
    return 0


def cmd_lessons(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    c = init_db()
    rows = c.execute("SELECT * FROM lessons ORDER BY id DESC LIMIT ?", (args.limit,)).fetchall()
    for r in rows:
        print(f"  #{r['cycle']:>3} [{r['kind']:>6}] {r['text']}")
    if not rows:
        print("no lessons yet -- run: python3 megon learn")
    return 0


def cmd_memory(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    c = init_db()
    store = Store(c)
    index = SemanticIndex(cfg, home_dir() / "index")
    index.load()
    retriever = Retriever(store, index, cfg)
    for h in retriever.search(" ".join(args.query), k=args.k):
        print(f"[{h['score']:.3f}] lex={h['lexical']} den={h['dense']} cov={h['coverage']} "
              f"chunk#{h['chunk_id']} doc#{h['doc_id']}")
        print(f"   {h['title'][:90]}")
        print(textwrap.fill(h["text"][:500], 96, initial_indent="   ", subsequent_indent="   "))
        print(f"   {h['url'][:96]}\n")
    return 0


def cmd_bandit(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    c = init_db()
    t = Tuner(c)
    print(f"{'arm':<20}{'pulls':>7}{'mean score':>12}   overrides")
    print("-" * 84)
    for name, ov in ARMS:
        s = t.state[name]
        print(f"{name:<20}{int(s['n']):>7}{s['mean']:>12.4f}   {json.dumps(ov)}")
    b, bov = t.best()
    print("-" * 84)
    print(f"current champion: {b} {json.dumps(bov)}")
    return 0


def cmd_export(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    p = export_sft(cfg, limit=args.limit)
    print(p)
    return 0


def cmd_finetune_script(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    p = home_dir() / "lora_finetune.py"
    p.write_text(FINETUNE_SCRIPT, encoding="utf-8")
    print(f"wrote {p}")
    print("run on a GPU box:  pip install transformers peft datasets accelerate bitsandbytes")
    print(f"                    python3 {p} --data megon_home/export/megon-sft-{today_key()}.jsonl")
    return 0


def cmd_daemon(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    sch = cfg.get("schedule", {})
    print(f"daemon: one cycle per day at {sch.get('hour',3):02d}:{sch.get('minute',15):02d} local "
          f"(budget {cfg['budget']['mb_per_day']} MB/day). Ctrl-C to stop.")
    stop = {"v": False}

    def sig(*a: Any) -> None:
        stop["v"] = True
        print("\ndaemon: stopping after current sleep")
    signal.signal(signal.SIGINT, sig)
    signal.signal(signal.SIGTERM, sig)
    while not stop["v"]:
        now = datetime.now()
        target = now.replace(hour=int(sch.get("hour", 3)), minute=int(sch.get("minute", 15)),
                             second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        wait = (target - now).total_seconds()
        log.i(f"sleeping {wait/3600:.2f} h until {target.strftime('%Y-%m-%d %H:%M')}")
        slept = 0.0
        while slept < wait and not stop["v"]:
            time.sleep(min(30.0, wait - slept))
            slept += 30.0
        if stop["v"]:
            break
        try:
            learn_cycle(cfg, verbose=args.verbose)
        except Exception as e:
            log.e(f"cycle failed: {e}\n{traceback.format_exc()[-600:]}")
    return 0


def cmd_policy_log(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    c = init_db()
    rows = c.execute("SELECT * FROM policy_log ORDER BY id DESC LIMIT ?", (args.limit,)).fetchall()
    for r in rows:
        print(f"  {r['ts']} {r['decision']:<6} {str(r['url'])[:70]}  :: {r['reason']}")
    if not rows:
        print("policy log empty (nothing blocked yet)")
    return 0


def cmd_reset(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    if not args.yes:
        print("this deletes the whole memory. re-run with --yes to confirm.")
        return 1
    for p in [db_path(), db_path().parent / "megon.db-wal", db_path().parent / "megon.db-shm"]:
        if p.exists():
            p.unlink()
    idx = home_dir() / "index"
    if idx.exists():
        shutil.rmtree(idx)
    print("memory wiped (config kept)")
    return 0


def cmd_serve(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    os.environ.setdefault("MEGON_HOME", str(home_dir()))
    from megon import web
    return web.main(argparse.Namespace(port=args.port, host=args.host))


def cmd_agent(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    from megon.agent import Agent, daemon
    if args.pause:
        pf = home_dir() / "PAUSE"
        pf.write_text(now_iso()) if args.pause == "on" else (pf.unlink(missing_ok=True))
        print(("PAUSED -- the kernel stops after its current step. "
               "Resume with: megon agent --pause off") if args.pause == "on"
              else "resumed.")
        return 0
    if args.policy:
        a = Agent(cfg)
        rows = a.policy_table()
        print(f"{'goal kind':<14}{'tool':<16}{'pulls':>7}{'mean reward':>13}")
        print("-" * 52)
        for r in rows:
            print(f"{r['kind']:<14}{r['tool']:<16}{r['n']:>7}{r['mean']:>13.4f}")
        if not rows:
            print("(nothing learned yet -- run: megon agent --steps 10)")
        return 0
    if args.log:
        a = Agent(cfg)
        for r in a.recent(args.log):
            print(f"  {r['ts']} [{r['goal_kind']:>11}] {r['tool']:<12} "
                  f"reward={r['reward']:<5} {str(r['arg'])[:60]}")
        return 0
    if args.forever:
        daemon(steps_per_wake=args.steps, sleep_s=args.sleep, cycle_every=args.cycle_every, cfg=cfg)
        return 0
    a = Agent(cfg)
    st = a.perceive()
    print(BANNER.format(v=VERSION))
    print(f"  memory   {st['memory']['docs']} docs · {st['memory']['chunks']} chunks · "
          f"{human(st['memory']['bytes'])}")
    print(f"  tools    {st['tools']} · verified skills {st['skills']['verified']}/{st['skills']['skills']}")
    print(f"  goals    {st['open_goals']} open · actions taken {st['actions_taken']}\n")
    out = a.run(args.steps, verbose=not args.quiet)
    ok = sum(1 for o in out if o.get("reward", 0) > 0)
    print(f"\n  {len(out)} steps · {ok} productive · policy arms "
          f"{len(a.policy_table())} · memory now {a.store.stats()['docs']} docs")
    return 0


def cmd_skill(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    from megon.skills import SkillLibrary
    c = init_db()
    lib = SkillLibrary(c, home_dir())
    if args.seed:
        print(f"installed {lib.install_seed()} starter skills")
    if args.run:
        name, _, rest = args.run.partition(":")
        arg = rest
        try:
            parsed = json.loads(arg) if arg.startswith("[") else [arg]
        except Exception:
            parsed = [arg]
        print(json.dumps(lib.run(name, *parsed), indent=2, ensure_ascii=False))
        return 0
    if args.invent:
        res = lib.invent_demos()
        for name, r in res.items():
            if r["ok"]:
                print(f"  ✓ {name:14s} {r['program']:34s} depth={r['depth']} "
                      f"(rejected {r['rejected_overfit']} overfit candidates)")
            else:
                print(f"  ✗ {name:14s} {r.get('stage')}: {r.get('reason', r.get('detail'))}")
        print(f"\n  {sum(1 for r in res.values() if r['ok'])}/{len(res)} invented and verified")
        return 0
    rows = lib.all()
    if not rows:
        print("no skills yet -- run: megon skill --seed")
        return 0
    print(f"{'skill':<18}{'ok':<5}{'family':<12}{'uses':>6}{'wins':>6}  purpose")
    print("-" * 92)
    for r in rows:
        print(f"{r['name']:<18}{'✓' if r['verified'] else '✗':<5}{(r['family'] or ''):<12}"
              f"{r['uses']:>6}{r['wins']:>6}  {(r['purpose'] or '')[:44]}")
    s = lib.stats()
    print("-" * 92)
    print(f"{s['verified']}/{s['skills']} verified · {s['uses']} uses · {s['wins']} wins")
    return 0


def cmd_evolve(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    from megon.evolve import Evolution
    e = Evolution(cfg)
    if args.history:
        rows = e.history(args.history)
        for r in rows:
            print(f"  {r['ts']} [{'+' if r['adopted'] else '-'}] {r['kind']:<12} "
                  f"{r['name']:<34} {str(r['detail'])[:60]}")
        if not rows:
            print("no self-development yet -- run: megon evolve --once")
        return 0
    if args.stats:
        s = e.stats()
        print(f"  grammar primitives : {s['grammar_size']}")
        print(f"  learned extensions : {s['extensions']}")
        print(f"  proposals / adopted: {s['proposals']} / {s['adopted']}")
        print(f"  verified skills    : {s['skills']}")
        return 0
    if args.commit:
        print(json.dumps(e.snapshot(), indent=2, ensure_ascii=False))
        return 0
    for i in range(args.rounds):
        r = e.evolve_once(commit=not args.no_commit)
        if not args.quiet:
            print(f"  round {i+1}: +{r['adopted']} primitives "
                  f"(grammar {r['grammar']}), skills {r['skills']}, "
                  f"gap-skill={r['gap_skill']}, committed={r['commit']}")
    return 0


def cmd_learn_code(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    """Read real open-source code and adopt what survives testing."""
    from megon.codex import learn_from_code
    from megon.skills import SkillLibrary
    c = init_db()
    store = Store(c)
    index = SemanticIndex(cfg, home_dir() / "index")
    index.load()
    retriever = Retriever(store, index, cfg)
    ctx = ToolCtx(cfg, c, store, retriever,
                  Answerer(store, retriever, cfg, MathSolver(kv_get(c, "solver_weights", None))))
    skills = SkillLibrary(c, home_dir())
    for q in args.query:
        r = learn_from_code(q, ctx, skills, max_repos=args.repos, verbose=not args.quiet)
        print(f"  {q!r}: scanned {r['repos']} repos, screened {r['screened']}, "
              f"adopted {r['adopted']}")
        for a in r["attributions"]:
            print(f"    + {a['function']:<24} from {a['repo']} [{a['license']}]")
    return 0


def cmd_bootstrap(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    """Give MEGON its starting syllabus before it develops curiosity."""
    from megon.tools import ToolCtx
    from megon.base import CURRICULUM, curriculum_topics
    c = init_db()
    store = Store(c)
    index = SemanticIndex(cfg, home_dir() / "index")
    index.load()
    retriever = Retriever(store, index, cfg)
    ctx = ToolCtx(cfg, c, store, retriever,
                  Answerer(store, retriever, cfg, MathSolver(kv_get(c, "solver_weights", None))))
    topics = curriculum_topics()
    if args.limit:
        topics = topics[:args.limit]
    print(f"  bootstrap: {len(topics)} topics across {len(CURRICULUM)} tracks")
    got = 0
    for track, items in CURRICULUM.items():
        for t in items:
            if t not in topics:
                continue
            res = call_tool("wikipedia", t, ctx)
            pages = res.get("pages", []) if isinstance(res, dict) else []
            for pg in pages:
                r = store.add_doc(pg["url"], pg["title"], pg["text"],
                                  f"bootstrap-{track}", "bootstrap")
                if r[0]:
                    got += 1
            if not args.quiet:
                print(f"    [{track:16}] {t[:44]:46} {len(pages)} pages")
    index.build(store, force=True)
    print(f"  bootstrap complete: {got} documents stored")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="megon", description="MEGON I AI - self-learning knowledge engine")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--log-level", default="info", choices=["debug", "info", "warn", "error"])
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("init", help="create home dir, database, config")
    sub.add_parser("status", help="show memory / index / budget state")

    s = sub.add_parser("learn", help="run one full self-learning cycle")
    s.add_argument("--mb", type=float, help="download ceiling for this cycle")
    s.add_argument("--docs", type=int, help="max documents for this cycle")
    s.add_argument("--minutes", type=float, help="wall-clock ceiling")
    s.add_argument("--no-tune", action="store_true")

    s = sub.add_parser("ask", help="ask one question")
    s.add_argument("question", nargs="*")

    s = sub.add_parser("chat", help="interactive REPL")
    s.add_argument("--mb", type=float); s.add_argument("--docs", type=int); s.add_argument("--minutes", type=float)

    s = sub.add_parser("eval", help="run / grow the benchmark")
    s.add_argument("--generate", type=int, default=0)
    s.add_argument("--limit", type=int, default=250)

    s = sub.add_parser("ingest", help="ingest URLs or local files")
    s.add_argument("targets", nargs="+")
    s.add_argument("--mb", type=float); s.add_argument("--docs", type=int); s.add_argument("--minutes", type=float)

    s = sub.add_parser("crawl", help="alias for ingest")
    s.add_argument("targets", nargs="+")
    s.add_argument("--mb", type=float); s.add_argument("--docs", type=int); s.add_argument("--minutes", type=float)

    sub.add_parser("leaderboard", help="cycle-over-cycle scores")
    sub.add_parser("sources", help="configured corpora + how much is in memory")
    sub.add_parser("bandit", help="self-tuner state")

    s = sub.add_parser("wishlist", help="weak topics queued for targeted crawling")
    s.add_argument("--add", nargs="*")

    s = sub.add_parser("lessons", help="reflection journal")
    s.add_argument("--limit", type=int, default=40)

    s = sub.add_parser("memory", help="raw retrieval over memory")
    s.add_argument("query", nargs="+")
    s.add_argument("--k", type=int, default=5)

    s = sub.add_parser("export-sft", help="write instruction data for real fine-tuning")
    s.add_argument("--limit", type=int, default=5000)

    sub.add_parser("finetune-script", help="write the LoRA fine-tuning script")

    s = sub.add_parser("daemon", help="run one cycle per day, forever")

    s = sub.add_parser("policy-log", help="audit trail")
    s.add_argument("--limit", type=int, default=40)

    s = sub.add_parser("reset", help="wipe memory")
    s.add_argument("--yes", action="store_true")

    s = sub.add_parser("agent", help="the self-deciding cognitive kernel")
    s.add_argument("--steps", type=int, default=10)
    s.add_argument("--forever", action="store_true", help="run unattended, 24/7")
    s.add_argument("--sleep", type=int, default=1800, help="seconds between wakes")
    s.add_argument("--cycle-every", type=int, default=8, help="learn cycle every N wakes")
    s.add_argument("--policy", action="store_true", help="show the learned action policy")
    s.add_argument("--log", type=int, default=0, help="show the last N actions")
    s.add_argument("--pause", choices=["on", "off"], help="stop / resume the kernel")
    s.add_argument("--quiet", action="store_true")

    s = sub.add_parser("skill", help="the self-verified skill library")
    s.add_argument("--seed", action="store_true", help="install the starter skills")
    s.add_argument("--run", default="", help="name:input, e.g. levenshtein:[\"kitten\",\"sitting\"]")
    s.add_argument("--invent", action="store_true",
                   help="synthesise new skills from examples alone (no LLM, no GPU)")

    s = sub.add_parser("bootstrap", help="learn the starting syllabus")
    s.add_argument("--limit", type=int, default=0, help="only the first N topics")
    s.add_argument("--quiet", action="store_true")

    s = sub.add_parser("learn-code", help="learn functions from real open-source repos")
    s.add_argument("query", nargs="+", help="topic to search for")
    s.add_argument("--repos", type=int, default=3)
    s.add_argument("--quiet", action="store_true")

    s = sub.add_parser("evolve", help="self-development: grow its own grammar and skills")
    s.add_argument("--rounds", type=int, default=1)
    s.add_argument("--stats", action="store_true")
    s.add_argument("--history", type=int, default=0)
    s.add_argument("--commit", action="store_true", help="git-commit the grown state now")
    s.add_argument("--no-commit", action="store_true")
    s.add_argument("--quiet", action="store_true")

    s = sub.add_parser("serve", help="launch the browser console")
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    s.add_argument("--host", default="0.0.0.0")
    return p


DISPATCH: Dict[str, Callable[[Dict[str, Any], argparse.Namespace], int]] = {
    "status": cmd_status, "learn": cmd_learn, "ask": cmd_ask, "chat": cmd_chat,
    "eval": cmd_eval, "ingest": cmd_ingest, "crawl": cmd_ingest, "leaderboard": cmd_leaderboard,
    "sources": cmd_sources, "wishlist": cmd_wishlist, "lessons": cmd_lessons,
    "memory": cmd_memory, "bandit": cmd_bandit, "export-sft": cmd_export,
    "finetune-script": cmd_finetune_script, "daemon": cmd_daemon, "policy-log": cmd_policy_log,
    "reset": cmd_reset, "serve": cmd_serve, "agent": cmd_agent, "skill": cmd_skill, "evolve": cmd_evolve,
        "learn-code": cmd_learn_code,
        "bootstrap": cmd_bootstrap,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    Log.level = args.log_level
    cfg = load_config()
    if args.cmd is None:
        parser.print_help()
        print("\nquick start:\n  python3 megon init\n  python3 megon learn --mb 4 --docs 20 --minutes 8"
              "\n  python3 megon ask \"what is a transformer?\"\n  python3 megon leaderboard\n")
        return 0
    if args.cmd == "init":
        c = init_db()
        save_config(cfg)
        Store(c).stats()
        SemanticIndex(cfg, home_dir() / "index").build(Store(c), force=False)
        print(BANNER.format(v=VERSION))
        print(f"  home      {home_dir()}")
        print(f"  database  {db_path()}")
        print(f"  config    {cfg_path()}")
        print(f"  sources   {len(cfg.get('sources', []))} corpora configured")
        print(f"  budget    {cfg['budget']['mb_per_day']} MB/day, "
              f"{cfg['budget']['max_docs_per_day']} docs/day, "
              f"≤{cfg['budget']['max_requests_per_min']} req/min\n"
              f"  next      python3 megon learn --mb 4 --docs 20 --minutes 8\n")
        return 0
    fn = DISPATCH.get(args.cmd)
    if fn is None:
        parser.print_help()
        return 2
    try:
        return fn(cfg, args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
