# Retrieval table (local evidence metrics)

Questions: `data/lme_s_raw/questions.jsonl`. Each row is a run directory under `runs/`.

| run | n | S-R@5 | S-R@20 | T-R@20 | mean tokens | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| bm25-lme_s_raw-seed0-warm | 470 | 0.838 | 0.904 | 0.753 | 1844.5 | 2.1 | 3.6 |
| bounded-lme_s_raw-seed0-warm | 470 | 0.893 | 0.947 | 0.924 | 1758.5 | 337.6 | 526.5 |
| dense-lme_s_raw-seed0-warm | 470 | 0.906 | 0.988 | 0.947 | 1569.7 | 347.9 | 497.3 |
| hybrid-lme_s_raw-seed0-warm | 470 | 0.893 | 0.966 | 0.885 | 1808.3 | 343.8 | 534.7 |
| one_hop-lme_s_raw-seed0-warm | 470 | 0.893 | 0.934 | 0.912 | 1753.4 | 370.0 | 550.8 |
| summary_first-lme_s_raw-seed0-warm | 470 | 0.841 | 0.879 | 0.862 | 1730.1 | 346.6 | 509.7 |


## Per question type: session recall@20

| run (session_recall@20) | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---|---|---|---|---|---|
| bm25-lme_s_raw-seed0-warm | 0.979 | 0.837 | 1.000 | 0.800 | 0.984 | 0.868 |
| bounded-lme_s_raw-seed0-warm | 0.993 | 0.943 | 1.000 | 0.800 | 1.000 | 0.910 |
| dense-lme_s_raw-seed0-warm | 1.000 | 0.981 | 1.000 | 1.000 | 1.000 | 0.973 |
| hybrid-lme_s_raw-seed0-warm | 1.000 | 0.952 | 1.000 | 0.967 | 1.000 | 0.928 |
| one_hop-lme_s_raw-seed0-warm | 0.986 | 0.914 | 1.000 | 0.767 | 1.000 | 0.902 |
| summary_first-lme_s_raw-seed0-warm | 0.944 | 0.874 | 0.964 | 0.733 | 0.859 | 0.854 |


## Per question type: turn recall@20

| run (turn_recall@20) | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---|---|---|---|---|---|
| bm25-lme_s_raw-seed0-warm | 0.910 | 0.604 | 0.875 | 0.372 | 0.930 | 0.754 |
| bounded-lme_s_raw-seed0-warm | 0.986 | 0.888 | 1.000 | 0.756 | 0.992 | 0.895 |
| dense-lme_s_raw-seed0-warm | 0.986 | 0.925 | 0.982 | 0.906 | 0.977 | 0.924 |
| hybrid-lme_s_raw-seed0-warm | 0.975 | 0.811 | 0.982 | 0.733 | 0.977 | 0.852 |
| one_hop-lme_s_raw-seed0-warm | 0.975 | 0.868 | 1.000 | 0.756 | 0.992 | 0.875 |
| summary_first-lme_s_raw-seed0-warm | 0.931 | 0.840 | 0.964 | 0.722 | 0.859 | 0.831 |
