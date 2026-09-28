# MEGON I AI :: cycle 0013

- **timestamp**: 2026-09-28T07:39:20Z
- **engine**: v1.0.0  |  **schema**: v3
- **composite score**: **0.5970**  (-0.0237 vs previous cycle)

## benchmark (self-growing)

| metric | value |
|---|---|
| n | 200 |
| score | 0.597 |
| cloze_acc | 0.6667 |
| qa_f1 | 0.0944 |
| qa_em | 0.0822 |
| retrieval_r1 | 1.0 |
| retrieval_r3 | 1.0 |
| math_acc | 0.0139 |
| math_bank_acc | 0.95 |
| coverage | 1.0 |

## memory

- documents: **1400** (3.9MB on disk)
- chunks: **6955**  |  words: **379.0K**  |  ~tokens: **504.1K**
- benchmark items: **1428**  |  lessons recorded: **1238**
- by source: `alpaca`=461, `fineweb-edu`=454, `wikitext-2`=324, `squad`=71, `agent-hf`=50, `wikipedia-targeted`=15, `agent-wikipedia`=11, `bootstrap-coding`=7

## self-tuning (UCB1 bandit over its own hyper-parameters)

| arm | pulls | mean score |
|---|---|---|
| base | 4 | 0.6125 |
| lexical_heavy | 4 | 0.6083 |
| bm25_long_docs | 4 | 0.6069 |
| dense_heavy | 4 | 0.6065 |
| rrf_soft | 3 | 0.6035 |
| wide_recall | 3 | 0.6021 |
| tight_precision | 4 | 0.6018 |
| rrf_sharp | 4 | 0.5998 |

## lessons learned this cycle

- **[skill]** math strategy 'ledger' win-rate 0.08 over 37 items -> prior now 0.30
- **[skill]** math strategy 'multiple' win-rate 0.08 over 13 items -> prior now 0.30
- **[skill]** math strategy 'percent_of' win-rate 0.00 over 6 items -> prior now 0.30
- **[skill]** math strategy 'rate_duration' win-rate 0.56 over 9 items -> prior now 0.77
- **[skill]** math strategy 'fraction_rest' win-rate 0.20 over 5 items -> prior now 0.30
- **[skill]** math strategy 'fraction_left' win-rate 0.33 over 3 items -> prior now 0.42
- **[skill]** math strategy 'group_each' win-rate 0.33 over 3 items -> prior now 0.30
- **[skill]** math strategy 'percent_left' win-rate 0.50 over 4 items -> prior now 0.44
- **[yield]** source 'wikipedia-transformer' produced 55 high-scoring answers
- **[yield]** source 'alpaca' produced 45 high-scoring answers
- **[yield]** source 'wikitext-2' produced 37 high-scoring answers
- **[yield]** source 'fineweb-edu' produced 28 high-scoring answers
- **[status]** cycle 13: score 0.597 (-0.024 vs last cycle)

## hardware cost of this cycle

- downloaded: **9.394 MB** across **153** documents
- stopped by: finished all sources
- politeness sleeps: 59.7 s (rate limiter, not CPU)
- phase seconds: acquire=76.09, consolidate=10.15, evaluate=12.65, tune=25.13, reflect=0.0

## honesty note

This card measures MEGON against **its own** growing benchmark. A rising curve here means its retrieval, coverage and arithmetic strategies are measurably improving. It does **not** mean MEGON beat any frontier model on any public leaderboard, and no amount of crawling would make it. The route from this to real weight updates is `megon export-sft` + `finetune-script` on a GPU machine.
