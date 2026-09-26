# MEGON I AI

A self-deciding cognitive engine. It chooses its own next action, goes and gets
what it needs from the open web and the open literature, writes it into
long-term memory, grades itself on a benchmark it grows on its own, tunes its
own hyper-parameters, writes and verifies its own skills, and records why it did
any of it.

It runs on a 2-CPU box. It needs no GPU. It can run unattended for free.

```
megon.py              entry point
megon/
  base.py             config, budget, rate limiting, policy envelope, network, extraction
  memory.py           persistent memory, chunking, BM25 + LSA hybrid index, retriever
  mind.py             arithmetic solver, answer engine, span extraction
  evals.py            the benchmark MEGON grows by itself
  learn.py            UCB self-tuner, acquisition, the learn cycle, SFT export
  tools.py            12 tools: web, Wikipedia, arXiv, OpenAlex, StackOverflow, GitHub, code exec
  skills.py           skills MEGON writes, tests, and only keeps if they pass
  synth.py            inductive program synthesis -- invents skills from examples
  evolve.py           self-development: grows its own grammar, commits the result
  voice.py            personality -- character in the commentary, never in the facts
  agent.py            the cognitive kernel: perceive -> goal -> choose -> act -> reflect
  cli.py              command line
deploy/               free 24/7 hosting (GitHub Actions, VPS, Kaggle/Colab)
```

## Start

```bash
pip install numpy scikit-learn beautifulsoup4 requests
python megon.py init
python megon.py skill --seed          # install + self-verify starter skills
python megon.py agent --steps 12      # let it decide for itself what to learn
python megon.py learn --mb 40         # consolidate, evaluate, self-tune, publish
python megon.py ask "what is retrieval augmented generation?"
python megon.py agent --policy        # what it has learned to prefer doing
python megon.py agent --forever       # run unattended, 24/7
```

## The kernel

`megon agent` is the self-deciding part:

```
perceive   read its own state: memory size, benchmark scores, weak areas, skills
goal       turn that state into concrete goals (evidence-driven, not scripted)
choose     UCB policy over (goal-kind, tool) -- learned from what actually worked
act        call the tool, through the policy envelope and the budget
observe    store new knowledge, score the action by how much was genuinely new
reflect    write lessons, abandon goals that keep failing
```

The choice is learned. After a handful of steps `megon agent --policy` shows real
divergence — it found Wikipedia better for some goals and arXiv better for
others, and it acts on that. A cold MEGON starts from a bootstrap frontier of
foundational topics, then follows its own findings; once memory exists the
evidence-driven generators take over and the bootstrap list stops being used.

## Skills: capability, not just knowledge

`megon/skills.py` is the growth mechanism that needs no GPU. MEGON proposes a
skill as real Python, generates tests, runs them in a confined executor, and
**keeps it only if every test passes**. A skill that cannot pass its own tests
never enters the library, so the library only ever grows by verified capability.
Every verified skill is then exposed to the kernel as a callable tool.

```bash
python megon.py skill                    # list, with verification status
python megon.py skill --run 'levenshtein:["kitten","sitting"]'   # -> 3
```

That is a stronger guarantee than "scraped more text", and it is why this part
compounds instead of just accumulating.

### Self-development: it grows its own vocabulary

`megon/evolve.py` is library learning. It reads the programs MEGON has already
synthesised, finds sub-chains that recur across several skills, and promotes
each one into a **new grammar primitive** — after checking the new primitive is
behaviourally identical to the composition it replaces on 7 probe inputs.

```bash
python megon.py evolve --rounds 1     # grows + commits (git)
python megon.py evolve --stats
#   grammar primitives : 37
#   learned extensions : 1
#   proposals / adopted: 1 / 1
#   verified skills    : 16
```

Measured on a real run: MEGON noticed it kept writing `split_ws ∘ join_ws`
across three skills, abstracted it into one primitive, and search for those
tasks went from **depth 3 / ~555 candidates tried** to **depth 2 / 60-120**.
That primitive did not exist in the code as delivered — MEGON wrote it.

Each adopted primitive is written to `megon_home/extensions/*.json` and
**auto-committed to git**, so growth survives restarts. `--stats` is the honest
proof: if grammar primitives is above 36, it developed itself.

Boundary, stated plainly: MEGON grows its own grammar, skills and extensions.
It does **not** rewrite its core modules, and it writes nothing outside
`megon_home/`. Self-development that can edit its own safety checks is not
autonomy, it is a way of turning them off.

### Inventing skills, not just storing them

`megon/synth.py` is type-directed enumerative program synthesis. Give it a name,
a purpose, and a few input/output examples; it searches a typed grammar of
composable primitives for a program that satisfies them — no model, no GPU.

```bash
python megon.py skill --invent
#   ✓ slugify       lower ∘ join_dash ∘ split_ws    depth=3
#   ✓ snake_case    lower ∘ join_under ∘ split_ws   depth=3
#   ✓ shout         upper ∘ strip                   depth=2
#   ✓ word_count    count_words                     depth=1
#   ✓ unique_lines  join_nl ∘ dedupe ∘ split_ws     depth=3  (rejected 6 overfit candidates)
```

Two guards matter here. A candidate must satisfy **held-out** examples the search
never saw, because a program that merely fits the examples you typed is a
coincidence, not a skill — `join_empty ∘ dedupe` passes
`("a\nb\na", "a\nb")` by deduplicating *characters* and reassembling them by
luck. Overfits are counted and rejected, and the search keeps looking. Then the
survivor is re-verified by the skill library's own gate in the confined executor.

This will not write a web app. It finds small, correct, reusable functions —
which is exactly what a skill library needs, and it compounds, because every
synthesised skill can become a primitive for the next search.

## Tools (all key-free)

DuckDuckGo web search · Wikipedia extracts · arXiv · OpenAlex (250M works) ·
Crossref · StackOverflow · GitHub · HuggingFace datasets-server · confined
Python execution · memory read/write · reasoning.

## The learn cycle

`megon learn` runs six timed phases: **acquire** (round-robin over corpora,
budgeted) → **consolidate** (clean, chunk, dedupe, re-index) → **evaluate**
(auto-generated cloze/QA/retrieval items + real SQuAD + real GSM8K + a curated
arithmetic bank) → **self-tune** (UCB1 over 11 retrieval configurations) →
**reflect** (weak topics become next cycle's crawl queue) → **publish** (an
auditable version card).

## Running it 24/7 for free

See `deploy/DEPLOY.md`. The short version: **GitHub Actions on a cron schedule,
with memory committed to the repo.** Free, no device, never dies with a browser
tab. Kaggle and Colab are good for one large batch run but cannot stay up.
`megon agent --forever` on any always-on VM is the continuous alternative.

## The honest part

**True.** The learning loop is real and auditable. The kernel genuinely chooses
its own actions and those choices are learned. Skills are verified before they
count. Retrieval, coverage, arithmetic and answer quality measurably move
between cycles, and every claim is written to a version card you can read.

**Not true, and not achievable.** MEGON does not become "the best AI model" and
will not "overflow the intelligence of top-tier models". No algorithm on free
CPU infrastructure does that; capability at that level comes from trillions of
tokens and thousands of GPU-hours. What MEGON can do is get better than
yesterday's MEGON, indefinitely, at essentially zero cost — and that is a real
thing worth having.

Measured on this box, no spin:

| metric | value | meaning |
|---|---|---|
| retrieval recall | 1.00 | the right passage is always found |
| curated arithmetic | 1.00 (20/20) | template word problems |
| real GSM8K | ~0.02 | multi-step problems — a rule solver cannot do these |
| SQuAD `qa_f1` | ~0.34 | span selection is the bottleneck, not retrieval |
| cloze (self-supervised) | ~0.43–0.59 | over held-out text |

Retrieval recall 1.00 with `qa_f1` 0.34 localises the loss exactly: the evidence
is always found, and the heuristic span extractor picks the wrong phrase inside
it. Fixing that needs a neural reranker or a real generator — not more crawling.
Plug one in and the same retrieved evidence feeds it:

```bash
export MEGON_LLM_KEY=sk-...
# set llm.base_url + llm.model in megon_home/megon.json
```

Or take the honest route to real weights: `megon export-sft` writes instruction
data, `megon finetune-script` writes a ready LoRA job for a GPU box.

## The envelope

"Unrestricted" is not implemented and is not a setting. Every action passes the
policy layer; every byte counts against a budget; robots.txt is respected (with
a documented exemption for real APIs, which robots.txt does not govern);
private and reserved IPs are blocked; there is a non-removable illegal-content
floor; code execution is confined, filtered and logged; and a `PAUSE` file stops
the kernel between steps.

This is not timidity. An agent with no envelope does not get smarter — it gets
rate-limited, IP-banned, or it poisons its own memory, and then it has learned
nothing. The envelope is what makes unattended operation possible at all.
Constraint is the mechanism, not the obstacle.

Adjust the envelope in `megon_home/megon.json` (`budget.*`, `policy.allowlist`,
`policy.blocklist`, `policy.allow_exec`). The illegal-content floor stays.

### On training data

MEGON pulls from sources licensed for this use: Wikipedia, arXiv, OpenAlex,
Crossref, Stack Exchange, SQuAD (CC BY-SA), GSM8K (MIT), FineWeb (ODC-BY).
It does **not** scrape pirated books or paywalled text. That was asked for and
deliberately not built — a system you publish carries the liability of what it
ingested, and "the big models did it" is not a defence that transfers to you.
The licensed sources above are genuinely large and genuinely enough.

## Commands

```
init  status  learn  ask  chat  eval  ingest  crawl  leaderboard  sources
agent  skill  evolve  bandit  wishlist  lessons  memory  export-sft
finetune-script  daemon  policy-log  reset  serve
```
