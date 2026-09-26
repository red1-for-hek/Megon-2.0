# MEGON I AI :: cycle 0005

- **timestamp**: 2026-09-26T23:48:25Z
- **engine**: v1.0.0  |  **schema**: v3
- **composite score**: **0.6276**  (+0.0187 vs previous cycle)

## benchmark (self-growing)

| metric | value |
|---|---|
| n | 200 |
| score | 0.6276 |
| cloze_acc | 0.7143 |
| qa_f1 | 0.1448 |
| qa_em | 0.137 |
| retrieval_r1 | 1.0 |
| retrieval_r3 | 1.0 |
| math_acc | 0.0139 |
| math_bank_acc | 1.0 |
| coverage | 1.0 |

## memory

- documents: **561** (2.2MB on disk)
- chunks: **3941**  |  words: **218.9K**  |  ~tokens: **291.1K**
- benchmark items: **560**  |  lessons recorded: **441**
- by source: `fineweb-edu`=166, `alpaca`=161, `wikitext-2`=109, `agent-hf`=50, `squad`=37, `wikipedia-targeted`=15, `agent-wikipedia`=9, `bootstrap-coding`=7

## self-tuning (UCB1 bandit over its own hyper-parameters)

| arm | pulls | mean score |
|---|---|---|
| lexical_heavy | 2 | 0.63 |
| base | 2 | 0.6271 |
| dense_heavy | 2 | 0.625 |
| rrf_soft | 1 | 0.6239 |
| bm25_long_docs | 1 | 0.6188 |
| bm25_short_docs | 1 | 0.6186 |
| wide_recall | 1 | 0.6181 |
| tight_precision | 2 | 0.6174 |

## lessons learned this cycle

- **[skill]** math strategy 'ledger' win-rate 0.09 over 33 items -> prior now 0.33
- **[skill]** math strategy 'multiple' win-rate 0.08 over 13 items -> prior now 0.30
- **[skill]** math strategy 'rate_duration' win-rate 0.56 over 9 items -> prior now 0.97
- **[skill]** math strategy 'group_each' win-rate 0.17 over 12 items -> prior now 0.40
- **[skill]** math strategy 'fraction_rest' win-rate 0.20 over 5 items -> prior now 0.57
- **[skill]** math strategy 'percent_left' win-rate 0.40 over 5 items -> prior now 0.76
- **[yield]** source 'wikipedia-transformer' produced 56 high-scoring answers
- **[yield]** source 'alpaca' produced 46 high-scoring answers
- **[yield]** source 'wikitext-2' produced 37 high-scoring answers
- **[yield]** source 'fineweb-edu' produced 30 high-scoring answers
- **[status]** cycle 5: score 0.628 (+0.019 vs last cycle)

## hardware cost of this cycle

- downloaded: **9.381 MB** across **148** documents
- stopped by: finished all sources
- politeness sleeps: 59.9 s (rate limiter, not CPU)
- phase seconds: acquire=79.8, consolidate=8.48, evaluate=12.13, tune=24.33, reflect=0.0

## honesty note

This card measures MEGON against **its own** growing benchmark. A rising curve here means its retrieval, coverage and arithmetic strategies are measurably improving. It does **not** mean MEGON beat any frontier model on any public leaderboard, and no amount of crawling would make it. The route from this to real weight updates is `megon export-sft` + `finetune-script` on a GPU machine.
