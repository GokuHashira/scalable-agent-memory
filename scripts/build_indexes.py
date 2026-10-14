"""Build and persist per-history indexes (cluster: --encoder bge; laptop smoke: --encoder stub)."""

from mem27.retrieval.indexing import main

if __name__ == "__main__":
    main()
