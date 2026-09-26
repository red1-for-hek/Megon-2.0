# MEGON I AI :: cycle 0004

- **timestamp**: 2026-09-26T21:30:58Z
- **engine**: v1.0.0  |  **schema**: v3
- **composite score**: **0.6089**  (-0.0361 vs previous cycle)

## benchmark (self-growing)

| metric | value |
|---|---|
| n | 200 |
| score | 0.6089 |
| cloze_acc | 0.6667 |
| qa_f1 | 0.1438 |
| qa_em | 0.1233 |
| retrieval_r1 | 1.0 |
| retrieval_r3 | 1.0 |
| math_acc | 0.0139 |
| math_bank_acc | 0.95 |
| coverage | 1.0 |

## memory

- documents: **454** (2.0MB on disk)
- chunks: **3575**  |  words: **199.4K**  |  ~tokens: **265.2K**
- benchmark items: **452**  |  lessons recorded: **334**
- by source: `fineweb-edu`=130, `alpaca`=126, `wikitext-2`=80, `agent-hf`=50, `squad`=30, `wikipedia-targeted`=15, `agent-wikipedia`=9, `bootstrap-coding`=7

## self-tuning (UCB1 bandit over its own hyper-parameters)

| arm | pulls | mean score |
|---|---|---|
| dense_heavy | 1 | 0.6502 |
| lexical_heavy | 1 | 0.6479 |
| base | 2 | 0.6271 |
| tight_precision | 1 | 0.6253 |
| rrf_soft | 1 | 0.6239 |
| bm25_long_docs | 1 | 0.6188 |
| bm25_short_docs | 1 | 0.6186 |
| wide_recall | 1 | 0.6181 |

## lessons learned this cycle

- **[skill]** math strategy 'ledger' win-rate 0.11 over 28 items -> prior now 0.42
- **[skill]** math strategy 'multiple' win-rate 0.08 over 13 items -> prior now 0.38
- **[skill]** math strategy 'percent_of' win-rate 0.00 over 6 items -> prior now 0.32
- **[skill]** math strategy 'rate_duration' win-rate 0.56 over 9 items -> prior now 1.00
- **[skill]** math strategy 'group_each' win-rate 0.17 over 12 items -> prior now 0.50
- **[skill]** math strategy 'fraction_of' win-rate 0.00 over 5 items -> prior now 0.56
- **[skill]** math strategy 'fraction_left' win-rate 0.33 over 3 items -> prior now 0.78
- **[skill]** math strategy 'percent_left' win-rate 0.50 over 4 items -> prior now 0.83
- **[gap]** weak on 'virgin mary allegedly appear 1858 lourdes' (1 failed items) -> queued for targeted crawl
- **[yield]** source 'wikipedia-transformer' produced 51 high-scoring answers
- **[yield]** source 'alpaca' produced 46 high-scoring answers
- **[yield]** source 'wikitext-2' produced 37 high-scoring answers
- **[yield]** source 'fineweb-edu' produced 31 high-scoring answers
- **[status]** cycle 4: score 0.609 (-0.036 vs last cycle)

## hardware cost of this cycle

- downloaded: **9.512 MB** across **126** documents
- stopped by: finished all sources
- politeness sleeps: 71.0 s (rate limiter, not CPU)
- phase seconds: acquire=88.78, consolidate=8.32, evaluate=12.44, tune=25.39, reflect=0.0

## honesty note

This card measures MEGON against **its own** growing benchmark. A rising curve here means its retrieval, coverage and arithmetic strategies are measurably improving. It does **not** mean MEGON beat any frontier model on any public leaderboard, and no amount of crawling would make it. The route from this to real weight updates is `megon export-sft` + `finetune-script` on a GPU machine.
