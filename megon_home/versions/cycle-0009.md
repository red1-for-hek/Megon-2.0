# MEGON I AI :: cycle 0009

- **timestamp**: 2026-09-27T17:42:36Z
- **engine**: v1.0.0  |  **schema**: v3
- **composite score**: **0.5866**  (-0.0109 vs previous cycle)

## benchmark (self-growing)

| metric | value |
|---|---|
| n | 200 |
| score | 0.5866 |
| cloze_acc | 0.619 |
| qa_f1 | 0.1277 |
| qa_em | 0.1233 |
| retrieval_r1 | 1.0 |
| retrieval_r3 | 1.0 |
| math_acc | 0.0139 |
| math_bank_acc | 0.9 |
| coverage | 1.0 |

## memory

- documents: **966** (3.0MB on disk)
- chunks: **5298**  |  words: **290.5K**  |  ~tokens: **386.4K**
- benchmark items: **997**  |  lessons recorded: **875**
- by source: `alpaca`=312, `fineweb-edu`=310, `wikitext-2`=197, `squad`=59, `agent-hf`=50, `wikipedia-targeted`=15, `agent-wikipedia`=9, `bootstrap-coding`=7

## self-tuning (UCB1 bandit over its own hyper-parameters)

| arm | pulls | mean score |
|---|---|---|
| lexical_heavy | 3 | 0.6174 |
| rrf_soft | 2 | 0.6173 |
| base | 3 | 0.6133 |
| bm25_long_docs | 3 | 0.6095 |
| wide_recall | 2 | 0.6094 |
| dense_heavy | 3 | 0.6092 |
| bm25_short_docs | 2 | 0.6089 |
| tight_precision | 3 | 0.607 |

## lessons learned this cycle

- **[skill]** math strategy 'ledger' win-rate 0.08 over 37 items -> prior now 0.30
- **[skill]** math strategy 'multiple' win-rate 0.08 over 13 items -> prior now 0.30
- **[skill]** math strategy 'percent_of' win-rate 0.00 over 6 items -> prior now 0.30
- **[skill]** math strategy 'rate_duration' win-rate 0.56 over 9 items -> prior now 0.86
- **[skill]** math strategy 'fraction_of' win-rate 0.00 over 5 items -> prior now 0.32
- **[skill]** math strategy 'percent_left' win-rate 0.40 over 5 items -> prior now 0.57
- **[skill]** math strategy 'group_each' win-rate 0.33 over 3 items -> prior now 0.30
- **[yield]** source 'wikipedia-transformer' produced 51 high-scoring answers
- **[yield]** source 'alpaca' produced 46 high-scoring answers
- **[yield]** source 'wikitext-2' produced 38 high-scoring answers
- **[yield]** source 'fineweb-edu' produced 29 high-scoring answers
- **[status]** cycle 9: score 0.587 (-0.011 vs last cycle)

## hardware cost of this cycle

- downloaded: **9.333 MB** across **147** documents
- stopped by: finished all sources
- politeness sleeps: 60.9 s (rate limiter, not CPU)
- phase seconds: acquire=77.61, consolidate=10.45, evaluate=19.23, tune=39.18, reflect=0.0

## honesty note

This card measures MEGON against **its own** growing benchmark. A rising curve here means its retrieval, coverage and arithmetic strategies are measurably improving. It does **not** mean MEGON beat any frontier model on any public leaderboard, and no amount of crawling would make it. The route from this to real weight updates is `megon export-sft` + `finetune-script` on a GPU machine.
