# MEGON I AI :: cycle 0006

- **timestamp**: 2026-09-27T01:54:04Z
- **engine**: v1.0.0  |  **schema**: v3
- **composite score**: **0.6340**  (+0.0064 vs previous cycle)

## benchmark (self-growing)

| metric | value |
|---|---|
| n | 200 |
| score | 0.634 |
| cloze_acc | 0.7619 |
| qa_f1 | 0.1241 |
| qa_em | 0.1096 |
| retrieval_r1 | 1.0 |
| retrieval_r3 | 1.0 |
| math_acc | 0.0139 |
| math_bank_acc | 1.0 |
| coverage | 1.0 |

## memory

- documents: **653** (2.4MB on disk)
- chunks: **4293**  |  words: **237.0K**  |  ~tokens: **315.3K**
- benchmark items: **668**  |  lessons recorded: **550**
- by source: `fineweb-edu`=202, `alpaca`=199, `wikitext-2`=119, `agent-hf`=50, `squad`=45, `wikipedia-targeted`=15, `agent-wikipedia`=9, `bootstrap-coding`=7

## self-tuning (UCB1 bandit over its own hyper-parameters)

| arm | pulls | mean score |
|---|---|---|
| lexical_heavy | 2 | 0.63 |
| base | 2 | 0.6271 |
| dense_heavy | 2 | 0.625 |
| bm25_long_docs | 2 | 0.6204 |
| wide_recall | 1 | 0.6181 |
| tight_precision | 2 | 0.6174 |
| rrf_soft | 2 | 0.6173 |
| rrf_sharp | 1 | 0.6121 |

## lessons learned this cycle

- **[skill]** math strategy 'ledger' win-rate 0.11 over 28 items -> prior now 0.30
- **[skill]** math strategy 'multiple' win-rate 0.08 over 13 items -> prior now 0.30
- **[skill]** math strategy 'percent_of' win-rate 0.00 over 6 items -> prior now 0.30
- **[skill]** math strategy 'rate_duration' win-rate 0.56 over 9 items -> prior now 0.94
- **[skill]** math strategy 'group_each' win-rate 0.17 over 12 items -> prior now 0.33
- **[skill]** math strategy 'fraction_rest' win-rate 0.20 over 5 items -> prior now 0.47
- **[skill]** math strategy 'fraction_left' win-rate 0.33 over 3 items -> prior now 0.69
- **[skill]** math strategy 'percent_left' win-rate 0.50 over 4 items -> prior now 0.72
- **[yield]** source 'wikipedia-transformer' produced 58 high-scoring answers
- **[yield]** source 'alpaca' produced 46 high-scoring answers
- **[yield]** source 'wikitext-2' produced 37 high-scoring answers
- **[yield]** source 'fineweb-edu' produced 30 high-scoring answers
- **[status]** cycle 6: score 0.634 (+0.006 vs last cycle)

## hardware cost of this cycle

- downloaded: **9.369 MB** across **132** documents
- stopped by: finished all sources
- politeness sleeps: 64.0 s (rate limiter, not CPU)
- phase seconds: acquire=79.69, consolidate=7.77, evaluate=10.1, tune=20.74, reflect=0.0

## honesty note

This card measures MEGON against **its own** growing benchmark. A rising curve here means its retrieval, coverage and arithmetic strategies are measurably improving. It does **not** mean MEGON beat any frontier model on any public leaderboard, and no amount of crawling would make it. The route from this to real weight updates is `megon export-sft` + `finetune-script` on a GPU machine.
