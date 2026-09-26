# MEGON I AI :: cycle 0002

- **timestamp**: 2026-09-26T15:02:54Z
- **engine**: v1.0.0  |  **schema**: v3
- **composite score**: **0.6264**  (-0.0238 vs previous cycle)

## benchmark (self-growing)

| metric | value |
|---|---|
| n | 200 |
| score | 0.6264 |
| cloze_acc | 0.7143 |
| qa_f1 | 0.169 |
| qa_em | 0.1507 |
| retrieval_r1 | 1.0 |
| retrieval_r3 | 1.0 |
| math_acc | 0.0139 |
| math_bank_acc | 0.95 |
| coverage | 1.0 |

## memory

- documents: **275** (1.5MB on disk)
- chunks: **2607**  |  words: **146.9K**  |  ~tokens: **195.3K**
- benchmark items: **236**  |  lessons recorded: **115**
- by source: `alpaca`=63, `fineweb-edu`=58, `wikitext-2`=55, `agent-hf`=50, `squad`=15, `wikipedia-targeted`=11, `agent-wikipedia`=9, `bootstrap-coding`=7

## self-tuning (UCB1 bandit over its own hyper-parameters)

| arm | pulls | mean score |
|---|---|---|
| base | 1 | 0.6502 |
| dense_heavy | 1 | 0.6502 |
| lexical_heavy | 1 | 0.6479 |
| tight_precision | 1 | 0.6253 |
| bm25_long_docs | 1 | 0.6188 |
| wide_recall | 1 | 0.6181 |
| bm25_short_docs | 0 | 0.0 |
| rrf_sharp | 0 | 0.0 |

## lessons learned this cycle

- **[skill]** math strategy 'ledger' win-rate 0.11 over 28 items -> prior now 0.67
- **[skill]** math strategy 'multiple' win-rate 0.08 over 13 items -> prior now 0.62
- **[skill]** math strategy 'percent_of' win-rate 0.00 over 6 items -> prior now 0.56
- **[skill]** math strategy 'rate_duration' win-rate 0.56 over 9 items -> prior now 1.05
- **[skill]** math strategy 'group_each' win-rate 0.17 over 12 items -> prior now 0.74
- **[skill]** math strategy 'fraction_of' win-rate 0.00 over 5 items -> prior now 0.75
- **[skill]** math strategy 'percent_left' win-rate 0.40 over 5 items -> prior now 0.93
- **[gap]** weak on 'basilica sacred heart notre dame beside' (1 failed items) -> queued for targeted crawl
- **[gap]** weak on 'scholastic magazine notre dame begin publishing' (1 failed items) -> queued for targeted crawl
- **[yield]** source 'wikipedia-transformer' produced 54 high-scoring answers
- **[yield]** source 'alpaca' produced 46 high-scoring answers
- **[yield]** source 'wikitext-2' produced 36 high-scoring answers
- **[yield]** source 'fineweb-edu' produced 34 high-scoring answers
- **[status]** cycle 2: score 0.626 (-0.024 vs last cycle)

## hardware cost of this cycle

- downloaded: **9.815 MB** across **143** documents
- stopped by: finished all sources
- politeness sleeps: 63.0 s (rate limiter, not CPU)
- phase seconds: acquire=83.17, consolidate=7.21, evaluate=11.11, tune=24.94, reflect=0.0

## honesty note

This card measures MEGON against **its own** growing benchmark. A rising curve here means its retrieval, coverage and arithmetic strategies are measurably improving. It does **not** mean MEGON beat any frontier model on any public leaderboard, and no amount of crawling would make it. The route from this to real weight updates is `megon export-sft` + `finetune-script` on a GPU machine.
