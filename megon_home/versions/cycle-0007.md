# MEGON I AI :: cycle 0007

- **timestamp**: 2026-09-27T07:14:42Z
- **engine**: v1.0.0  |  **schema**: v3
- **composite score**: **0.6247**  (-0.0093 vs previous cycle)

## benchmark (self-growing)

| metric | value |
|---|---|
| n | 200 |
| score | 0.6247 |
| cloze_acc | 0.7619 |
| qa_f1 | 0.1142 |
| qa_em | 0.1233 |
| retrieval_r1 | 1.0 |
| retrieval_r3 | 1.0 |
| math_acc | 0.0139 |
| math_bank_acc | 0.95 |
| coverage | 1.0 |

## memory

- documents: **757** (2.6MB on disk)
- chunks: **4605**  |  words: **253.3K**  |  ~tokens: **336.9K**
- benchmark items: **779**  |  lessons recorded: **658**
- by source: `alpaca`=243, `fineweb-edu`=238, `wikitext-2`=136, `squad`=52, `agent-hf`=50, `wikipedia-targeted`=15, `agent-wikipedia`=9, `bootstrap-coding`=7

## self-tuning (UCB1 bandit over its own hyper-parameters)

| arm | pulls | mean score |
|---|---|---|
| lexical_heavy | 2 | 0.63 |
| base | 2 | 0.6271 |
| dense_heavy | 2 | 0.625 |
| bm25_long_docs | 2 | 0.6204 |
| tight_precision | 2 | 0.6174 |
| rrf_soft | 2 | 0.6173 |
| wide_recall | 2 | 0.6094 |
| bm25_short_docs | 2 | 0.6089 |

## lessons learned this cycle

- **[skill]** math strategy 'ledger' win-rate 0.11 over 28 items -> prior now 0.30
- **[skill]** math strategy 'multiple' win-rate 0.08 over 13 items -> prior now 0.30
- **[skill]** math strategy 'percent_of' win-rate 0.00 over 6 items -> prior now 0.30
- **[skill]** math strategy 'rate_duration' win-rate 0.56 over 9 items -> prior now 0.91
- **[skill]** math strategy 'group_each' win-rate 0.17 over 12 items -> prior now 0.30
- **[skill]** math strategy 'fraction_of' win-rate 0.00 over 5 items -> prior now 0.42
- **[skill]** math strategy 'percent_left' win-rate 0.40 over 5 items -> prior now 0.66
- **[yield]** source 'wikipedia-transformer' produced 55 high-scoring answers
- **[yield]** source 'alpaca' produced 44 high-scoring answers
- **[yield]** source 'wikitext-2' produced 37 high-scoring answers
- **[yield]** source 'fineweb-edu' produced 30 high-scoring answers
- **[status]** cycle 7: score 0.625 (-0.009 vs last cycle)

## hardware cost of this cycle

- downloaded: **9.355 MB** across **145** documents
- stopped by: finished all sources
- politeness sleeps: 63.2 s (rate limiter, not CPU)
- phase seconds: acquire=77.43, consolidate=8.75, evaluate=12.43, tune=26.09, reflect=0.0

## honesty note

This card measures MEGON against **its own** growing benchmark. A rising curve here means its retrieval, coverage and arithmetic strategies are measurably improving. It does **not** mean MEGON beat any frontier model on any public leaderboard, and no amount of crawling would make it. The route from this to real weight updates is `megon export-sft` + `finetune-script` on a GPU machine.
