# Retrieval table (local evidence metrics, G)

Questions: `data/lme_s/questions.jsonl`. Each row is a run directory under `runs/`.

| run | n | S-R@5 | S-R@20 | T-R@20 | mean tokens | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| bm25-lme_s-seed0-warm | 470 | 0.838 | 0.904 | 0.753 | 1844.5 | 2.1 | 3.6 |
| dense-lme_s-seed0-warm | 470 | 0.906 | 0.988 | 0.947 | 1569.7 | 3395.6 | 5146.5 |


## Per question type: session recall@20

| run (session_recall@20) | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---|---|---|---|---|---|
| bm25-lme_s-seed0-warm | 0.979 | 0.837 | 1.000 | 0.800 | 0.984 | 0.868 |
| dense-lme_s-seed0-warm | 1.000 | 0.981 | 1.000 | 1.000 | 1.000 | 0.973 |


## Per question type: turn recall@20

| run (turn_recall@20) | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---|---|---|---|---|---|
| bm25-lme_s-seed0-warm | 0.910 | 0.604 | 0.875 | 0.372 | 0.930 | 0.754 |
| dense-lme_s-seed0-warm | 0.986 | 0.925 | 0.982 | 0.906 | 0.977 | 0.924 |
