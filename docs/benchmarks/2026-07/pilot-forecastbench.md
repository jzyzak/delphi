# Evaluation (3 questions)

## Proper scores
- brier: 0.0842 (95% CI [0.0000, 0.2500])
- log: 0.2483 (95% CI [0.0000, 0.6931])

## Per-domain (brier)
- infer: 0.1250 (n=2)
- polymarket: 0.0025 (n=1)

## Baseline deltas (brier; negative = model beats baseline)
- vs market_consensus: -0.0963

## Reliability
| bin | n | mean_pred | mean_outcome | gap |
| --- | --- | --- | --- | --- |
| [0.00, 0.10) | 2 | 0.025 | 0.000 | 0.025 |
| [0.50, 0.60) | 1 | 0.500 | 0.000 | 0.500 |

ECE=0.1835  MCE=0.5000  n=3
2026-07-14 20:06:59 [info     ] leakage_rate_estimated         aggregate_rate=0.0 flagged=0 slice_id= total=3
leakage_rate=0.0000
clean_fraction (flagged-at-chance robustness)=1.0000
flagged=0/3
  supervisor: 0/3 (0.000)
