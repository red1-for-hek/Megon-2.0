# MEGON I AI :: cycle 0010

- **timestamp**: 2026-09-27T20:26:08Z
- **engine**: v1.0.0  |  **schema**: v3
- **composite score**: **0.6050**  (+0.0184 vs previous cycle)

## benchmark (self-growing)

| metric | value |
|---|---|
| n | 200 |
| score | 0.605 |
| cloze_acc | 0.6667 |
| qa_f1 | 0.1277 |
| qa_em | 0.1233 |
| retrieval_r1 | 1.0 |
| retrieval_r3 | 1.0 |
| math_acc | 0.0139 |
| math_bank_acc | 0.95 |
| coverage | 1.0 |

## memory

- documents: **1087** (3.2MB on disk)
- chunks: **5776**  |  words: **315.9K**  |  ~tokens: **420.1K**
- benchmark items: **1104**  |  lessons recorded: **984**
- by source: `alpaca`=355, `fineweb-edu`=346, `wikitext-2`=235, `squad`=62, `agent-hf`=50, `wikipedia-targeted`=15, `agent-wikipedia`=10, `bootstrap-coding`=7

## self-tuning (UCB1 bandit over its own hyper-parameters)

| arm | pulls | mean score |
|---|---|---|
| lexical_heavy | 3 | 0.6174 |
| base | 3 | 0.6133 |
| bm25_long_docs | 3 | 0.6095 |
| dense_heavy | 3 | 0.6092 |
| tight_precision | 3 | 0.607 |
| rrf_sharp | 2 | 0.6061 |
| rrf_soft | 3 | 0.6035 |
| wide_recall | 3 | 0.6021 |

## lessons learned this cycle

- **[skill]** math strategy 'ledger' win-rate 0.08 over 37 items -> prior now 0.30
- **[skill]** math strategy 'multiple' win-rate 0.08 over 13 items -> prior now 0.30
- **[skill]** math strategy 'percent_of' win-rate 0.00 over 6 items -> prior now 0.30
- **[skill]** math strategy 'rate_duration' win-rate 0.56 over 9 items -> prior now 0.84
- **[skill]** math strategy 'fraction_rest' win-rate 0.20 over 5 items -> prior now 0.33
- **[skill]** math strategy 'fraction_left' win-rate 0.33 over 3 items -> prior now 0.54
- **[skill]** math strategy 'group_each' win-rate 0.33 over 3 items -> prior now 0.30
- **[skill]** math strategy 'percent_left' win-rate 0.50 over 4 items -> prior now 0.54
- **[yield]** source 'wikipedia-transformer' produced 51 high-scoring answers
- **[yield]** source 'alpaca' produced 45 high-scoring answers
- **[yield]** source 'wikitext-2' produced 37 high-scoring answers
- **[yield]** source 'fineweb-edu' produced 29 high-scoring answers
- **[status]** cycle 10: score 0.605 (+0.018 vs last cycle)

## hardware cost of this cycle

- downloaded: **9.451 MB** across **165** documents
- stopped by: finished all sources
- politeness sleeps: 69.3 s (rate limiter, not CPU)
- phase seconds: acquire=93.2, consolidate=9.83, evaluate=13.13, tune=27.0, reflect=0.0

## honesty note

This card measures MEGON against **its own** growing benchmark. A rising curve here means its retrieval, coverage and arithmetic strategies are measurably improving. It does **not** mean MEGON beat any frontier model on any public leaderboard, and no amount of crawling would make it. The route from this to real weight updates is `megon export-sft` + `finetune-script` on a GPU machine.
