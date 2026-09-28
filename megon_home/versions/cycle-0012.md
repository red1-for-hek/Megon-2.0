# MEGON I AI :: cycle 0012

- **timestamp**: 2026-09-28T01:41:07Z
- **engine**: v1.0.0  |  **schema**: v3
- **composite score**: **0.6207**  (+0.0084 vs previous cycle)

## benchmark (self-growing)

| metric | value |
|---|---|
| n | 200 |
| score | 0.6207 |
| cloze_acc | 0.7619 |
| qa_f1 | 0.0977 |
| qa_em | 0.0822 |
| retrieval_r1 | 1.0 |
| retrieval_r3 | 1.0 |
| math_acc | 0.0139 |
| math_bank_acc | 0.95 |
| coverage | 1.0 |

## memory

- documents: **1292** (3.7MB on disk)
- chunks: **6599**  |  words: **360.1K**  |  ~tokens: **478.9K**
- benchmark items: **1320**  |  lessons recorded: **1165**
- by source: `alpaca`=421, `fineweb-edu`=418, `wikitext-2`=295, `squad`=68, `agent-hf`=50, `wikipedia-targeted`=15, `agent-wikipedia`=11, `bootstrap-coding`=7

## self-tuning (UCB1 bandit over its own hyper-parameters)

| arm | pulls | mean score |
|---|---|---|
| base | 4 | 0.6125 |
| dense_heavy | 3 | 0.6092 |
| lexical_heavy | 4 | 0.6083 |
| tight_precision | 3 | 0.607 |
| bm25_long_docs | 4 | 0.6069 |
| rrf_sharp | 3 | 0.6041 |
| rrf_soft | 3 | 0.6035 |
| wide_recall | 3 | 0.6021 |

## lessons learned this cycle

- **[skill]** math strategy 'ledger' win-rate 0.08 over 37 items -> prior now 0.30
- **[skill]** math strategy 'multiple' win-rate 0.08 over 13 items -> prior now 0.30
- **[skill]** math strategy 'percent_of' win-rate 0.00 over 6 items -> prior now 0.30
- **[skill]** math strategy 'rate_duration' win-rate 0.56 over 9 items -> prior now 0.79
- **[skill]** math strategy 'fraction_rest' win-rate 0.20 over 5 items -> prior now 0.30
- **[skill]** math strategy 'fraction_left' win-rate 0.33 over 3 items -> prior now 0.48
- **[skill]** math strategy 'group_each' win-rate 0.33 over 3 items -> prior now 0.30
- **[skill]** math strategy 'percent_left' win-rate 0.50 over 4 items -> prior now 0.47
- **[yield]** source 'wikipedia-transformer' produced 59 high-scoring answers
- **[yield]** source 'alpaca' produced 45 high-scoring answers
- **[yield]** source 'wikitext-2' produced 37 high-scoring answers
- **[yield]** source 'fineweb-edu' produced 28 high-scoring answers
- **[status]** cycle 12: score 0.621 (+0.008 vs last cycle)

## hardware cost of this cycle

- downloaded: **9.38 MB** across **136** documents
- stopped by: finished all sources
- politeness sleeps: 57.0 s (rate limiter, not CPU)
- phase seconds: acquire=77.73, consolidate=11.03, evaluate=19.19, tune=39.34, reflect=0.01

## honesty note

This card measures MEGON against **its own** growing benchmark. A rising curve here means its retrieval, coverage and arithmetic strategies are measurably improving. It does **not** mean MEGON beat any frontier model on any public leaderboard, and no amount of crawling would make it. The route from this to real weight updates is `megon export-sft` + `finetune-script` on a GPU machine.
