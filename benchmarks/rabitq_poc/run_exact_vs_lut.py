"""
Compare Rust LUT kernel vs Rust exact-math kernel on the same index.

Both paths run inside the same Rust pipeline — same rotation, same Lloyd-Max
codebook, same encoded codes, same per-vector scale. The only difference is
how the inner product is computed:

    - search()       : bit-plane SIMD popcount through u8 LUTs + calibration
    - search_exact() : float32 x_hat reconstruction + BLAS matmul

If they differ by more than ~0.1pp on recall, that gap is precisely the
recall cost of the LUT-quantization noise in the SIMD kernel. Apples-to-
apples at the implementation level.
"""

import json
import os
import time

import h5py
import numpy as np
from turbovec import TurboQuantIndex

DATA_DIR = os.path.expanduser("~/data/py-turboquant")
HERE = os.path.dirname(__file__)
SEED = 42
N_DB = 100_000
K = 64
K_VALUES = [1, 2, 4, 8, 16, 32, 64]


def load_openai(dim):
    all_vecs = np.load(os.path.join(DATA_DIR, f"openai-{dim}.npy"))
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(all_vecs))
    db = all_vecs[idx[:N_DB]].astype(np.float32)
    q = all_vecs[idx[N_DB : N_DB + 1_000]].astype(np.float32)
    db /= np.linalg.norm(db, axis=-1, keepdims=True)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return db, q, dim


def load_glove():
    with h5py.File(os.path.join(DATA_DIR, "glove-200-angular.hdf5"), "r") as f:
        all_train = f["train"][:].astype(np.float32)
        queries = f["test"][:].astype(np.float32)
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(all_train), N_DB, replace=False)
    db = all_train[idx]
    db /= np.linalg.norm(db, axis=-1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True)
    return db, queries, 200


DATASETS = {
    "glove-200":   load_glove,
    "openai-1536": lambda: load_openai(1536),
    "openai-3072": lambda: load_openai(3072),
}


def recall_at_1_at_k(true_top1, predicted_indices, k):
    return float(np.mean([true_top1[i] in predicted_indices[i, :k] for i in range(len(true_top1))]))


def run(dataset, bits):
    print(f"\n=== {dataset}, {bits}-bit ===")
    t0 = time.time()
    database, queries, dim = DATASETS[dataset]()
    true_top1 = np.argmax(queries @ database.T, axis=1)
    print(f"  data + ground truth: {time.time() - t0:.1f}s")

    t0 = time.time()
    index = TurboQuantIndex(dim, bit_width=bits)
    index.add(database)
    index.prepare()
    print(f"  build + prepare: {time.time() - t0:.1f}s")

    t0 = time.time()
    _, lut_indices = index.search(queries, k=K)
    lut_indices = np.array(lut_indices)
    lut_recalls = {str(k): round(recall_at_1_at_k(true_top1, lut_indices, k), 4) for k in K_VALUES}
    print(f"  LUT kernel:   recall@1 = {lut_recalls['1']:.4f} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    _, exact_indices = index.search_exact(queries, k=K)
    exact_indices = np.array(exact_indices)
    exact_recalls = {str(k): round(recall_at_1_at_k(true_top1, exact_indices, k), 4) for k in K_VALUES}
    print(f"  exact kernel: recall@1 = {exact_recalls['1']:.4f} ({time.time() - t0:.1f}s)")
    print(f"  Δ (exact - LUT) recall@1 = {exact_recalls['1'] - lut_recalls['1']:+.4f}")

    return {
        "dataset": dataset, "dim": dim, "bit_width": bits, "seed": SEED,
        "n_db": N_DB, "n_queries": len(queries),
        "lut_kernel": lut_recalls,
        "exact_kernel": exact_recalls,
    }


if __name__ == "__main__":
    results = {}
    for dataset in ["glove-200", "openai-1536", "openai-3072"]:
        for bits in (2, 4):
            results[f"{dataset}_{bits}bit"] = run(dataset, bits)

    out_path = os.path.join(HERE, "exact_vs_lut.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_path}")

    print("\n" + "=" * 72)
    print(f"{'cell':<22} {'LUT':>10} {'exact':>10} {'Δ exact-LUT':>14}")
    print("-" * 72)
    for key, r in results.items():
        lut = r["lut_kernel"]["1"]
        exact = r["exact_kernel"]["1"]
        print(f"{key:<22} {lut:>10.4f} {exact:>10.4f} {exact-lut:>+14.4f}")
