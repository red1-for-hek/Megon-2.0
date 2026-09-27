# MEGON I AI :: cycle 0011

- **timestamp**: 2026-09-27T23:12:54Z
- **engine**: v1.0.0  |  **schema**: v3
- **composite score**: **0.6123**  (+0.0073 vs previous cycle)

## benchmark (self-growing)

| metric | value |
|---|---|
| n | 200 |
| score | 0.6123 |
| cloze_acc | 0.7143 |
| qa_f1 | 0.1105 |
| qa_em | 0.1096 |
| retrieval_r1 | 1.0 |
| retrieval_r3 | 1.0 |
| math_acc | 0.0139 |
| math_bank_acc | 0.95 |
| coverage | 1.0 |

## memory

- documents: **1201** (3.5MB on disk)
- chunks: **6244**  |  words: **341.2K**  |  ~tokens: **453.8K**
- benchmark items: **1212**  |  lessons recorded: **1092**
- by source: `alpaca`=391, `fineweb-edu`=382, `wikitext-2`=273, `squad`=65, `agent-hf`=50, `wikipedia-targeted`=15, `agent-wikipedia`=11, `bootstrap-coding`=7

## self-tuning (UCB1 bandit over its own hyper-parameters)

| arm | pulls | mean score |
|---|---|---|
| lexical_heavy | 3 | 0.6174 |
| base | 3 | 0.6133 |
| bm25_long_docs | 3 | 0.6095 |
| dense_heavy | 3 | 0.6092 |
| tight_precision | 3 | 0.607 |
| rrf_sharp | 3 | 0.6041 |
| rrf_soft | 3 | 0.6035 |
| wide_recall | 3 | 0.6021 |

## lessons learned this cycle

- **[skill]** math strategy 'ledger' win-rate 0.08 over 37 items -> prior now 0.30
- **[skill]** math strategy 'multiple' win-rate 0.08 over 13 items -> prior now 0.30
- **[skill]** math strategy 'percent_of' win-rate 0.00 over 6 items -> prior now 0.30
- **[skill]** math strategy 'rate_duration' win-rate 0.56 over 9 items -> prior now 0.82
- **[skill]** math strategy 'fraction_rest' win-rate 0.20 over 5 items -> prior now 0.30
- **[skill]** math strategy 'percent_left' win-rate 0.40 over 5 items -> prior now 0.49
- **[skill]** math strategy 'group_each' win-rate 0.33 over 3 items -> prior now 0.30
- **[yield]** source 'wikipedia-transformer' produced 55 high-scoring answers
- **[yield]** source 'alpaca' produced 49 high-scoring answers
- **[yield]** source 'wikitext-2' produced 37 high-scoring answers
- **[yield]** source 'fineweb-edu' produced 29 high-scoring answers
- **[status]** cycle 11: score 0.612 (+0.007 vs last cycle)

## hardware cost of this cycle

- downloaded: **9.414 MB** across **158** documents
- stopped by: finished all sources
- politeness sleeps: 69.1 s (rate limiter, not CPU)
- phase seconds: acquire=84.43, consolidate=8.87, evaluate=10.6, tune=21.24, reflect=0.0

## honesty note

This card measures MEGON against **its own** growing benchmark. A rising curve here means its retrieval, coverage and arithmetic strategies are measurably improving. It does **not** mean MEGON beat any frontier model on any public leaderboard, and no amount of crawling would make it. The route from this to real weight updates is `megon export-sft` + `finetune-script` on a GPU machine.
