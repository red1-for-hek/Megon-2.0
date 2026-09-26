# MEGON I AI :: cycle 0003

- **timestamp**: 2026-09-26T18:36:03Z
- **engine**: v1.0.0  |  **schema**: v3
- **composite score**: **0.6450**  (+0.0186 vs previous cycle)

## benchmark (self-growing)

| metric | value |
|---|---|
| n | 200 |
| score | 0.645 |
| cloze_acc | 0.7619 |
| qa_f1 | 0.1698 |
| qa_em | 0.1507 |
| retrieval_r1 | 1.0 |
| retrieval_r3 | 1.0 |
| math_acc | 0.0139 |
| math_bank_acc | 1.0 |
| coverage | 1.0 |

## memory

- documents: **368** (1.7MB on disk)
- chunks: **2981**  |  words: **166.6K**  |  ~tokens: **221.6K**
- benchmark items: **344**  |  lessons recorded: **224**
- by source: `fineweb-edu`=94, `alpaca`=89, `wikitext-2`=78, `agent-hf`=50, `squad`=22, `wikipedia-targeted`=12, `agent-wikipedia`=9, `bootstrap-coding`=7

## self-tuning (UCB1 bandit over its own hyper-parameters)

| arm | pulls | mean score |
|---|---|---|
| base | 1 | 0.6502 |
| dense_heavy | 1 | 0.6502 |
| lexical_heavy | 1 | 0.6479 |
| tight_precision | 1 | 0.6253 |
| rrf_soft | 1 | 0.6239 |
| bm25_long_docs | 1 | 0.6188 |
| bm25_short_docs | 1 | 0.6186 |
| wide_recall | 1 | 0.6181 |

## lessons learned this cycle

- **[skill]** math strategy 'ledger' win-rate 0.11 over 28 items -> prior now 0.53
- **[skill]** math strategy 'multiple' win-rate 0.08 over 13 items -> prior now 0.48
- **[skill]** math strategy 'percent_of' win-rate 0.00 over 6 items -> prior now 0.42
- **[skill]** math strategy 'rate_duration' win-rate 0.56 over 9 items -> prior now 1.02
- **[skill]** math strategy 'group_each' win-rate 0.17 over 12 items -> prior now 0.61
- **[skill]** math strategy 'fraction_rest' win-rate 0.20 over 5 items -> prior now 0.69
- **[skill]** math strategy 'fraction_left' win-rate 0.33 over 3 items -> prior now 0.88
- **[skill]** math strategy 'percent_left' win-rate 0.50 over 4 items -> prior now 0.88
- **[yield]** source 'wikipedia-transformer' produced 58 high-scoring answers
- **[yield]** source 'alpaca' produced 46 high-scoring answers
- **[yield]** source 'wikitext-2' produced 37 high-scoring answers
- **[yield]** source 'fineweb-edu' produced 33 high-scoring answers
- **[status]** cycle 3: score 0.645 (+0.019 vs last cycle)

## hardware cost of this cycle

- downloaded: **9.39 MB** across **134** documents
- stopped by: finished all sources
- politeness sleeps: 63.5 s (rate limiter, not CPU)
- phase seconds: acquire=79.77, consolidate=7.64, evaluate=12.23, tune=25.07, reflect=0.01

## honesty note

This card measures MEGON against **its own** growing benchmark. A rising curve here means its retrieval, coverage and arithmetic strategies are measurably improving. It does **not** mean MEGON beat any frontier model on any public leaderboard, and no amount of crawling would make it. The route from this to real weight updates is `megon export-sft` + `finetune-script` on a GPU machine.
