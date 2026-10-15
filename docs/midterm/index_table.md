# Index build cost

| component | histories | build s | disk MB | peak RSS MB | build @100,000 | disk @100,000 | build @1,000,000 | disk @1,000,000 |
|---|---|---|---|---|---|---|---|---|
| bm25 | 500 | 25.5 | 281.5 | 3547 | 10.4 s | 114 MB | 103.5 s | 1142 MB |
| dense | 500 | 1009.0 | 512.4 | 3547 | 409.2 s | 208 MB | 4091.6 s | 2078 MB |
| summary_first | 500 | 507.2 | 181.5 | 3549 | 205.7 s | 74 MB | 2056.7 s | 736 MB |
| links | 500 | 46.7 | 450.8 | 3549 | 19.0 s | 183 MB | 189.5 s | 1828 MB |

Measured on 246,607 turn notes. Projection is linear in turn-note count (per-history flat indexes, no cross-history effects); it does not model approximate (HNSW) or Lucene indexes at 1M scale. Peak RSS is the process high-water mark, not a per-component delta.
