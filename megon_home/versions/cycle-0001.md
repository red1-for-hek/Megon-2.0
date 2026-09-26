# MEGON I AI :: cycle 0001

- **timestamp**: 2026-09-26T14:12:04Z
- **engine**: v1.0.0  |  **schema**: v3
- **composite score**: **0.6502**  (first measured cycle)

## benchmark (self-growing)

| metric | value |
|---|---|
| n | 128 |
| score | 0.6502 |
| cloze_acc | 0.7143 |
| qa_f1 | 0.2331 |
| qa_em | 0.1892 |
| retrieval_r1 | 1.0 |
| retrieval_r3 | 1.0 |
| math_acc | 0.0278 |
| math_bank_acc | 1.0 |
| coverage | 1.0 |

## memory

- documents: **165** (764.2KB on disk)
- chunks: **1317**  |  words: **71.5K**  |  ~tokens: **95.2K**
- benchmark items: **128**  |  lessons recorded: **39**
- by source: `agent-hf`=50, `alpaca`=38, `wikitext-2`=32, `fineweb-edu`=22, `squad`=8, `bootstrap-coding`=7, `agent-wikipedia`=6, `wikipedia-llm`=1

## self-tuning (UCB1 bandit over its own hyper-parameters)

| arm | pulls | mean score |
|---|---|---|
| base | 1 | 0.6502 |
| dense_heavy | 1 | 0.6502 |
| lexical_heavy | 1 | 0.6479 |
| wide_recall | 0 | 0.0 |
| tight_precision | 0 | 0.0 |
| bm25_long_docs | 0 | 0.0 |
| bm25_short_docs | 0 | 0.0 |
| rrf_sharp | 0 | 0.0 |

## lessons learned this cycle

- **[skill]** math strategy 'ledger' win-rate 0.25 over 12 items -> prior now 0.85
- **[skill]** math strategy 'multiple' win-rate 0.10 over 10 items -> prior now 0.79
- **[skill]** math strategy 'percent_of' win-rate 0.00 over 4 items -> prior now 0.75
- **[skill]** math strategy 'fraction_rest' win-rate 0.20 over 5 items -> prior now 0.83
- **[skill]** math strategy 'rate_duration' win-rate 0.83 over 6 items -> prior now 1.08
- **[skill]** math strategy 'group_each' win-rate 0.40 over 5 items -> prior now 0.91
- **[skill]** math strategy 'percent_left' win-rate 0.67 over 3 items -> prior now 1.02
- **[gap]** weak on 'front notre dame main building' (1 failed items) -> queued for targeted crawl
- **[gap]** weak on 'grotto notre dame' (1 failed items) -> queued for targeted crawl
- **[gap]** weak on 'sits top main building notre dame' (1 failed items) -> queued for targeted crawl
- **[gap]** weak on 'often notre dame's juggler published' (1 failed items) -> queued for targeted crawl
- **[gap]** weak on 'daily student paper notre dame called' (1 failed items) -> queued for targeted crawl
- **[yield]** source 'wikipedia-transformer' produced 54 high-scoring answers
- **[yield]** source 'alpaca' produced 40 high-scoring answers

## hardware cost of this cycle

- downloaded: **9.533 MB** across **191** documents
- stopped by: finished all sources
- politeness sleeps: 43.8 s (rate limiter, not CPU)
- phase seconds: acquire=60.57, consolidate=6.62, evaluate=6.05, tune=24.23, reflect=0.0

## honesty note

This card measures MEGON against **its own** growing benchmark. A rising curve here means its retrieval, coverage and arithmetic strategies are measurably improving. It does **not** mean MEGON beat any frontier model on any public leaderboard, and no amount of crawling would make it. The route from this to real weight updates is `megon export-sft` + `finetune-script` on a GPU machine.
