# MEGON I AI :: cycle 0008

- **timestamp**: 2026-09-27T13:04:53Z
- **engine**: v1.0.0  |  **schema**: v3
- **composite score**: **0.5975**  (-0.0272 vs previous cycle)

## benchmark (self-growing)

| metric | value |
|---|---|
| n | 200 |
| score | 0.5975 |
| cloze_acc | 0.6667 |
| qa_f1 | 0.0964 |
| qa_em | 0.0959 |
| retrieval_r1 | 1.0 |
| retrieval_r3 | 1.0 |
| math_acc | 0.0139 |
| math_bank_acc | 0.95 |
| coverage | 1.0 |

## memory

- documents: **864** (2.8MB on disk)
- chunks: **5023**  |  words: **275.7K**  |  ~tokens: **366.7K**
- benchmark items: **889**  |  lessons recorded: **767**
- by source: `alpaca`=279, `fineweb-edu`=274, `wikitext-2`=167, `squad`=56, `agent-hf`=50, `wikipedia-targeted`=15, `agent-wikipedia`=9, `bootstrap-coding`=7

## self-tuning (UCB1 bandit over its own hyper-parameters)

| arm | pulls | mean score |
|---|---|---|
| dense_heavy | 2 | 0.625 |
| bm25_long_docs | 2 | 0.6204 |
| lexical_heavy | 3 | 0.6174 |
| tight_precision | 2 | 0.6174 |
| rrf_soft | 2 | 0.6173 |
| base | 3 | 0.6133 |
| wide_recall | 2 | 0.6094 |
| bm25_short_docs | 2 | 0.6089 |

## lessons learned this cycle

- **[skill]** math strategy 'ledger' win-rate 0.08 over 37 items -> prior now 0.30
- **[skill]** math strategy 'multiple' win-rate 0.08 over 13 items -> prior now 0.30
- **[skill]** math strategy 'percent_of' win-rate 0.00 over 6 items -> prior now 0.30
- **[skill]** math strategy 'rate_duration' win-rate 0.56 over 9 items -> prior now 0.89
- **[skill]** math strategy 'fraction_rest' win-rate 0.20 over 5 items -> prior now 0.39
- **[skill]** math strategy 'fraction_left' win-rate 0.33 over 3 items -> prior now 0.61
- **[skill]** math strategy 'group_each' win-rate 0.33 over 3 items -> prior now 0.30
- **[skill]** math strategy 'percent_left' win-rate 0.50 over 4 items -> prior now 0.62
- **[yield]** source 'wikipedia-transformer' produced 56 high-scoring answers
- **[yield]** source 'alpaca' produced 46 high-scoring answers
- **[yield]** source 'wikitext-2' produced 37 high-scoring answers
- **[yield]** source 'fineweb-edu' produced 29 high-scoring answers
- **[status]** cycle 8: score 0.598 (-0.027 vs last cycle)

## hardware cost of this cycle

- downloaded: **9.414 MB** across **151** documents
- stopped by: finished all sources
- politeness sleeps: 64.8 s (rate limiter, not CPU)
- phase seconds: acquire=84.19, consolidate=12.33, evaluate=26.15, tune=53.52, reflect=0.0

## honesty note

This card measures MEGON against **its own** growing benchmark. A rising curve here means its retrieval, coverage and arithmetic strategies are measurably improving. It does **not** mean MEGON beat any frontier model on any public leaderboard, and no amount of crawling would make it. The route from this to real weight updates is `megon export-sft` + `finetune-script` on a GPU machine.
