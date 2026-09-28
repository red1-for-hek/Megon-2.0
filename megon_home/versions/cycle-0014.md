# MEGON I AI :: cycle 0014

- **timestamp**: 2026-09-28T15:57:44Z
- **engine**: v1.0.0  |  **schema**: v3
- **composite score**: **0.6087**  (+0.0117 vs previous cycle)

## benchmark (self-growing)

| metric | value |
|---|---|
| n | 200 |
| score | 0.6087 |
| cloze_acc | 0.7143 |
| qa_f1 | 0.0955 |
| qa_em | 0.0959 |
| retrieval_r1 | 1.0 |
| retrieval_r3 | 1.0 |
| math_acc | 0.0139 |
| math_bank_acc | 0.95 |
| coverage | 1.0 |

## memory

- documents: **1503** (4.2MB on disk)
- chunks: **7441**  |  words: **404.2K**  |  ~tokens: **537.5K**
- benchmark items: **1536**  |  lessons recorded: **1310**
- by source: `alpaca`=496, `fineweb-edu`=490, `wikitext-2`=353, `squad`=74, `agent-hf`=50, `wikipedia-targeted`=15, `agent-wikipedia`=11, `bootstrap-coding`=7

## self-tuning (UCB1 bandit over its own hyper-parameters)

| arm | pulls | mean score |
|---|---|---|
| base | 4 | 0.6125 |
| lexical_heavy | 4 | 0.6083 |
| bm25_long_docs | 4 | 0.6069 |
| dense_heavy | 4 | 0.6065 |
| wide_recall | 4 | 0.6031 |
| rrf_soft | 4 | 0.6024 |
| tight_precision | 4 | 0.6018 |
| rrf_sharp | 4 | 0.5998 |

## lessons learned this cycle

- **[skill]** math strategy 'ledger' win-rate 0.08 over 37 items -> prior now 0.30
- **[skill]** math strategy 'multiple' win-rate 0.08 over 13 items -> prior now 0.30
- **[skill]** math strategy 'percent_of' win-rate 0.00 over 6 items -> prior now 0.30
- **[skill]** math strategy 'rate_duration' win-rate 0.56 over 9 items -> prior now 0.75
- **[skill]** math strategy 'fraction_rest' win-rate 0.20 over 5 items -> prior now 0.30
- **[skill]** math strategy 'percent_left' win-rate 0.40 over 5 items -> prior now 0.40
- **[skill]** math strategy 'group_each' win-rate 0.33 over 3 items -> prior now 0.30
- **[yield]** source 'wikipedia-transformer' produced 56 high-scoring answers
- **[yield]** source 'alpaca' produced 45 high-scoring answers
- **[yield]** source 'wikitext-2' produced 36 high-scoring answers
- **[yield]** source 'fineweb-edu' produced 30 high-scoring answers
- **[status]** cycle 14: score 0.609 (+0.012 vs last cycle)

## hardware cost of this cycle

- downloaded: **9.446 MB** across **148** documents
- stopped by: finished all sources
- politeness sleeps: 59.0 s (rate limiter, not CPU)
- phase seconds: acquire=79.76, consolidate=10.59, evaluate=12.43, tune=26.1, reflect=0.0

## honesty note

This card measures MEGON against **its own** growing benchmark. A rising curve here means its retrieval, coverage and arithmetic strategies are measurably improving. It does **not** mean MEGON beat any frontier model on any public leaderboard, and no amount of crawling would make it. The route from this to real weight updates is `megon export-sft` + `finetune-script` on a GPU machine.
