# Evaluation (160 questions)

## Proper scores
- brier: 0.2056 (95% CI [0.1816, 0.2310])
- log: 0.6787 (95% CI [0.5341, 0.8414])

## Per-domain (brier)
- acled: 0.1069 (n=24)
- dbnomics: 0.2324 (n=26)
- fred: 0.2362 (n=16)
- infer: 0.2532 (n=8)
- manifold: 0.3018 (n=9)
- metaculus: 0.1291 (n=14)
- polymarket: 0.2605 (n=20)
- wikipedia: 0.1574 (n=22)
- yfinance: 0.2520 (n=21)

## Baseline deltas (brier; negative = model beats baseline)
- vs market_consensus: -0.1338

## Reliability
| bin | n | mean_pred | mean_outcome | gap |
| --- | --- | --- | --- | --- |
| [0.00, 0.10) | 40 | 0.001 | 0.100 | 0.099 |
| [0.20, 0.30) | 3 | 0.274 | 0.000 | 0.274 |
| [0.50, 0.60) | 117 | 0.522 | 0.547 | 0.025 |

ECE=0.0484  MCE=0.2743  n=160
leakage_rate=0.0000
clean_fraction (flagged-at-chance robustness)=1.0000
flagged=0/160
  supervisor: 0/160 (0.000)
