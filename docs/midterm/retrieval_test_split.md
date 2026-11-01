# Retrieval on the 400-question test split

Questions: the test split of `data/lme_s/questions.jsonl` (stratified by question type and abstention, seed 0), matched by question id to the runs built on `data/lme_s_raw/questions.jsonl`. Abstention questions have no gold evidence and are excluded from recall.

Scored test questions: 376 (of 400; 24 abstention). Session recall uses 376 questions with gold sessions; turn recall uses 376 with gold turns.

Values are means with 95% bootstrap CIs (10,000 resamples of questions, seed 0).

| method | S-R@5 | S-R@20 | T-R@20 | mean tokens | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| dense | 0.915 [0.893, 0.935] | 0.989 [0.982, 0.995] | 0.953 [0.934, 0.969] | 1579 | 348.8 | 489.7 |
| hybrid | 0.905 [0.882, 0.927] | 0.968 [0.954, 0.980] | 0.891 [0.864, 0.917] | 1805 | 344.0 | 536.1 |
| bounded | 0.905 [0.882, 0.927] | 0.957 [0.940, 0.972] | 0.935 [0.914, 0.955] | 1752 | 337.7 | 526.6 |
| one_hop | 0.905 [0.882, 0.927] | 0.943 [0.924, 0.961] | 0.922 [0.899, 0.943] | 1746 | 369.8 | 563.9 |
| bm25 | 0.845 [0.814, 0.875] | 0.913 [0.889, 0.935] | 0.763 [0.723, 0.801] | 1843 | 2.1 | 3.7 |
| summary_first | 0.855 [0.825, 0.883] | 0.886 [0.858, 0.912] | 0.868 [0.839, 0.896] | 1718 | 346.9 | 506.3 |

## Paired differences against dense

Each cell is the mean per-question difference (method minus reference) with its 95% CI. Verdicts use the pre-registered ±0.02 margin: *equivalent* when the CI lies inside ±0.02, *better*/*worse* when it excludes 0, otherwise *inconclusive*.

| method | ΔS-R@5 | ΔS-R@20 | ΔT-R@20 |
|---|---|---|---|
| bm25 | -0.069 [-0.096, -0.044] worse | -0.077 [-0.099, -0.055] worse | -0.190 [-0.228, -0.152] worse |
| bounded | -0.010 [-0.027, +0.006] inconclusive | -0.032 [-0.048, -0.018] worse | -0.017 [-0.038, +0.002] inconclusive |
| hybrid | -0.010 [-0.027, +0.006] inconclusive | -0.022 [-0.033, -0.011] worse | -0.062 [-0.087, -0.037] worse |
| one_hop | -0.010 [-0.027, +0.006] inconclusive | -0.046 [-0.064, -0.029] worse | -0.030 [-0.052, -0.009] worse |
| summary_first | -0.060 [-0.084, -0.037] worse | -0.103 [-0.130, -0.078] worse | -0.084 [-0.113, -0.057] worse |

## Per question type: session_recall@20

| method | knowledge-update (n=57) | multi-session (n=97) | single-session-assistant (n=45) | single-session-preference (n=24) | single-session-user (n=51) | temporal-reasoning (n=102) |
|---|---|---|---|---|---|---|
| bm25 | 0.974 | 0.863 | 1.000 | 0.875 | 0.980 | 0.863 |
| bounded | 0.991 | 0.950 | 1.000 | 0.875 | 1.000 | 0.924 |
| dense | 1.000 | 0.985 | 1.000 | 1.000 | 1.000 | 0.975 |
| hybrid | 1.000 | 0.955 | 1.000 | 1.000 | 1.000 | 0.923 |
| one_hop | 0.982 | 0.920 | 1.000 | 0.833 | 1.000 | 0.916 |
| summary_first | 0.965 | 0.873 | 0.956 | 0.792 | 0.863 | 0.858 |

## Per question type: turn_recall@20

| method | knowledge-update (n=57) | multi-session (n=97) | single-session-assistant (n=45) | single-session-preference (n=24) | single-session-user (n=51) | temporal-reasoning (n=102) |
|---|---|---|---|---|---|---|
| bm25 | 0.895 | 0.625 | 0.867 | 0.465 | 0.951 | 0.750 |
| bounded | 0.982 | 0.905 | 1.000 | 0.819 | 0.990 | 0.908 |
| dense | 0.982 | 0.934 | 0.978 | 0.944 | 0.990 | 0.925 |
| hybrid | 0.968 | 0.825 | 0.978 | 0.806 | 0.990 | 0.842 |
| one_hop | 0.968 | 0.883 | 1.000 | 0.819 | 0.990 | 0.890 |
| summary_first | 0.947 | 0.842 | 0.956 | 0.778 | 0.863 | 0.834 |
