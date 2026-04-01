"""
Benchmark py-turboquant against the paper's datasets.

Reproduces Section 4.4 of TurboQuant (arXiv:2504.19874):
  - GloVe d=200: 100K vectors, 10K queries
  - OpenAI DBpedia d=1536: 100K vectors, 1K queries
  - OpenAI DBpedia d=3072: 100K vectors, 1K queries

Usage:
  python3 benchmark.py glove
  python3 benchmark.py openai-1536
  python3 benchmark.py openai-3072
  python3 benchmark.py glove openai-1536 openai-3072
"""

import os
import sys
import time

import h5py
import numpy as np

from turbovec import TurboQuantIndex

DATA_DIR = os.path.expanduser("~/data/py-turboquant")

GLOVE_PATH = os.path.join(DATA_DIR, "glove-200-angular.hdf5")
GLOVE_URL = "http://ann-benchmarks.com/glove-200-angular.hdf5"


def download_glove():
    if os.path.exists(GLOVE_PATH):
        print(f"  Already downloaded: {GLOVE_PATH}")
        return GLOVE_PATH

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"  Downloading {GLOVE_URL}...")
    import subprocess
    subprocess.run(["curl", "-L", "-o", GLOVE_PATH, GLOVE_URL], check=True)
    print(f"  Saved: {GLOVE_PATH} ({os.path.getsize(GLOVE_PATH) / 1024 / 1024:.0f} MB)")
    return GLOVE_PATH


def load_glove(seed=42):
    if not os.path.exists(GLOVE_PATH):
        download_glove()
    f = h5py.File(GLOVE_PATH, "r")
    all_train = f["train"][:].astype(np.float32)
    queries = f["test"][:].astype(np.float32)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(all_train), 100_000, replace=False)
    database = all_train[idx]
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True)
    return database, queries


def download_openai(dim=1536):
    from datasets import load_dataset

    path = os.path.join(DATA_DIR, f"openai-{dim}.npy")
    if os.path.exists(path):
        print(f"  Already downloaded: {path}")
        return path

    os.makedirs(DATA_DIR, exist_ok=True)
    name = f"Qdrant/dbpedia-entities-openai3-text-embedding-3-large-{dim}-1M"
    col = f"text-embedding-3-large-{dim}-embedding"
    print(f"  Downloading {name}...")
    ds = load_dataset(name, split="train")
    ds.set_format("numpy")
    vecs = ds[col].astype(np.float32)
    np.save(path, vecs)
    print(f"  Saved: {path} ({os.path.getsize(path) / 1024 / 1024:.0f} MB)")
    return path


def load_openai(dim=1536, seed=42):
    path = os.path.join(DATA_DIR, f"openai-{dim}.npy")
    if not os.path.exists(path):
        download_openai(dim)

    all_vecs = np.load(path)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_vecs))
    database = all_vecs[idx[:100_000]]
    queries = all_vecs[idx[100_000:101_000]]
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True)
    return database, queries


def recall_at_1_at_k(true_top1, predicted_indices, k):
    return np.mean([true_top1[i] in predicted_indices[i, :k] for i in range(len(true_top1))])


def bench_search(index, queries, k, n_runs=1):
    # Warmup
    index.search(queries[:1], k)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _, indices = index.search(queries, k)
        times.append(time.perf_counter() - t0)
    median_time = sorted(times)[len(times) // 2]
    return median_time, indices


def run_benchmark(database, queries, bit_widths, label=""):
    n, dim = database.shape
    n_queries = len(queries)

    # Ground truth
    true_top1 = np.argmax(queries @ database.T, axis=1)

    # Encode + search each bit width
    results = {}
    for bw in bit_widths:
        t0 = time.time()
        index = TurboQuantIndex(database.shape[1], bit_width=bw)
        index.add(database)
        encode_time = time.time() - t0

        index.write("/tmp/bench.tq")
        file_size = os.path.getsize("/tmp/bench.tq")
        loaded = TurboQuantIndex.load("/tmp/bench.tq")

        search_time, all_indices = bench_search(loaded, queries, k=64)

        recalls = {}
        for k in [1, 2, 4, 8, 16, 32, 64]:
            recalls[k] = recall_at_1_at_k(true_top1, all_indices, k)

        results[bw] = {
            "encode_time": encode_time,
            "search_time": search_time,
            "file_size": file_size,
            "recalls": recalls,
        }

    # Print summary table
    original_mb = n * dim * 4 / 1024 / 1024
    bw_labels = [f"{bw}-bit" for bw in bit_widths]
    header = f"  {'k':>4}  " + "  ".join(f"{l:>10}" for l in bw_labels)
    print(header)
    print(f"  {'─' * 4}  " + "  ".join("─" * 10 for _ in bit_widths))
    for k in [1, 2, 4, 8, 16, 32, 64]:
        row = f"  {k:>4}  " + "  ".join(f"{results[bw]['recalls'][k]:>10.4f}" for bw in bit_widths)
        print(row)

    # Summary table
    print()
    header = f"  {'':>7}  " + "  ".join(f"{bw}-bit" for bw in bit_widths)
    print(header)
    print(f"  {'':>7}  " + "  ".join("─" * 10 for _ in bit_widths))
    print(f"  {'Size':>7}  " + "  ".join(f"{results[bw]['file_size'] / 1024 / 1024:>7.1f} MB" for bw in bit_widths))
    print(f"  {'Comp.':>7}  " + "  ".join(f"{original_mb / (results[bw]['file_size'] / 1024 / 1024):>8.1f}x" for bw in bit_widths))
    print(f"  {'Index':>7}  " + "  ".join(f"{results[bw]['encode_time']*1000:>7.0f} ms" for bw in bit_widths))
    n_threads = os.environ.get("RAYON_NUM_THREADS", "all")
    print(f"  {'Search':>7}  " + "  ".join(f"{results[bw]['search_time'] / n_queries * 1000:>5.3f}ms/q" for bw in bit_widths) + f"  ({n_threads} threads)")


def run_add_benchmark(database, bit_widths, label=""):
    n, dim = database.shape
    print(f"\nadd benchmark ({label})")

    for bw in bit_widths:
        # Build index from first 99K
        index = TurboQuantIndex(dim, bit_width=bw)
        index.add(database[:n - 100])

        # Add 1 vector
        v = database[n - 1:n]
        t0 = time.time()
        index.add(v)
        time_1 = time.time() - t0

        # Fresh index, then add 100 vectors one at a time
        index = TurboQuantIndex(dim, bit_width=bw)
        index.add(database[:n - 100])
        t0 = time.time()
        for i in range(100):
            index.add(database[n - 100 + i : n - 100 + i + 1])
        time_100 = time.time() - t0

        # Verify correctness: same results as building all at once
        index_full = TurboQuantIndex(dim, bit_width=bw)
        index_full.add(database)
        q = database[:1]  # use first vector as query
        _, i_inc = index.search(q, k=5)
        _, i_full = index_full.search(q, k=5)
        correct = np.array_equal(i_inc, i_full)

        print(f"  {bw}-bit: add 1 vector = {time_1*1000:.2f}ms, "
              f"add 100 vectors (1 at a time) = {time_100*1000:.0f}ms "
              f"({time_100/100*1000:.2f}ms/vec), "
              f"correct={correct}")


DATASETS = {
    "glove": ("GloVe d=200 (100K vectors, 10K queries)", load_glove),
    "openai-1536": ("OpenAI DBpedia d=1536 (100K vectors, 1K queries)", lambda: load_openai(1536)),
    "openai-3072": ("OpenAI DBpedia d=3072 (100K vectors, 1K queries)", lambda: load_openai(3072)),
}

if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else ["glove"]

    # Handle download command
    if args[0] == "download":
        targets = args[1:] if len(args) > 1 else ["glove", "openai-1536", "openai-3072"]
        for t in targets:
            if t == "glove":
                download_glove()
            elif t == "openai-1536":
                download_openai(1536)
            elif t == "openai-3072":
                download_openai(3072)
            else:
                print(f"Unknown download target: {t}")
        sys.exit(0)

    print("py-turboquant benchmark")
    print("=" * 60)

    for name in args:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}")
            continue

        label, loader = DATASETS[name]
        print(f"\nDataset: {label}")
        database, queries = loader()
        n, dim = database.shape
        print(f"Database: {n:,} x {dim}, Queries: {len(queries):,}")
        print(f"Original size: {n * dim * 4 / 1024 / 1024:.1f} MB (FP32)")

        run_benchmark(database, queries, [2, 4], name)
        run_add_benchmark(database, [2, 4], name)

    print("\nDone.")
