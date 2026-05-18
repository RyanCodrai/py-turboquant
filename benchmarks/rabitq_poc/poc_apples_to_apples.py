"""
Apples-to-apples POC: consumes Rust's EXACT rotation matrix, Lloyd-Max
boundaries, and centroids — exported by `cargo run --example dump_state` —
so the only thing left that can differ from the real Rust pipeline is the
scoring kernel (numpy f32 matmul vs Rust bit-plane SIMD).

If the corrected recall here matches the Rust prototype's recall to within
the SIMD kernel noise floor, that proves the correction math is implemented
identically in both pipelines.

If they DON'T match, that's a real signal something is wrong in one of the
two paths.

Run prerequisites:
    cargo run -p turbovec --example dump_state --release -- benchmarks/rabitq_poc/rust_state
"""

import json
import os
import time

import h5py
import numpy as np

DATA_DIR = os.path.expanduser("~/data/py-turboquant")
HERE = os.path.dirname(__file__)
STATE_DIR = os.path.join(HERE, "rust_state")
SEED = 42
N_DB = 100_000
K_VALUES = [1, 2, 4, 8, 16, 32, 64]


def load_rust_state(dim, bits):
    """Load rotation + boundaries + centroids dumped by examples/dump_state.rs."""
    path = os.path.join(STATE_DIR, f"state_d{dim}_b{bits}.bin")
    raw = np.fromfile(path, dtype="<f4")
    n_rot = dim * dim
    n_bnd = (1 << bits) - 1
    n_cent = 1 << bits
    assert len(raw) == n_rot + n_bnd + n_cent, f"size mismatch in {path}: {len(raw)} vs {n_rot + n_bnd + n_cent}"
    rotation = raw[:n_rot].reshape(dim, dim).copy()
    boundaries = raw[n_rot : n_rot + n_bnd].copy()
    centroids = raw[n_rot + n_bnd :].copy()
    return rotation, boundaries, centroids


def encode(vectors, rotation, boundaries, centroids):
    norms = np.linalg.norm(vectors, axis=1).astype(np.float32)
    unit = vectors / np.clip(norms[:, None], 1e-10, None)
    rotated = unit @ rotation.T
    codes_idx = np.searchsorted(boundaries, rotated)
    x_hat = centroids[codes_idx]
    return norms, rotated, x_hat


def compute_scales(rotated, x_hat, norms):
    inner = np.einsum("nd,nd->n", rotated, x_hat)
    inner = np.clip(inner, 1e-10, None)
    return {
        "baseline": norms,
        "form_B_paper": norms / inner,
    }


def score_and_topk(query_rot, x_hat_db, scalars, k):
    raw = x_hat_db @ query_rot.T
    scored = raw * scalars[:, None]
    topk = np.argpartition(-scored, k, axis=0)[:k]
    topk_scores = np.take_along_axis(scored, topk, axis=0)
    order = np.argsort(-topk_scores, axis=0)
    return np.take_along_axis(topk, order, axis=0).T


def recall_at_1_at_k(true_top1, predicted, k):
    return float(np.mean([true_top1[i] in predicted[i, :k] for i in range(len(true_top1))]))


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


def run(dataset_label, bits):
    dim = {"glove-200": 200, "openai-1536": 1536, "openai-3072": 3072}[dataset_label]
    print(f"\n=== {dataset_label}, {bits}-bit (apples-to-apples vs Rust) ===")

    t0 = time.time()
    database, queries, _ = DATASETS[dataset_label]()
    true_top1 = np.argmax(queries @ database.T, axis=1)
    print(f"  data + ground truth: {time.time() - t0:.1f}s")

    rotation, boundaries, centroids = load_rust_state(dim, bits)
    print(f"  loaded Rust state for d={dim} bits={bits}")
    print(f"    rotation[:3,:3] = {rotation[:3,:3]}")
    print(f"    centroids = {centroids}")

    norms, rotated, x_hat = encode(database, rotation, boundaries, centroids)
    print(f"  ||x_hat|| mean = {np.linalg.norm(x_hat, axis=1).mean():.4f}")

    scalars = compute_scales(rotated, x_hat, norms)
    queries_rot = (queries @ rotation.T).astype(np.float32)

    out = {}
    k_max = max(K_VALUES)
    for name, s in scalars.items():
        top = score_and_topk(queries_rot, x_hat, s, k_max)
        recalls = {str(k): round(recall_at_1_at_k(true_top1, top, k), 4) for k in K_VALUES}
        out[name] = recalls
        print(f"  {name:<14} recall@1 = {recalls['1']:.4f}")
    return out


if __name__ == "__main__":
    results = {}
    for dataset in ["glove-200", "openai-1536", "openai-3072"]:
        for bits in (2, 4):
            results[f"{dataset}_{bits}bit"] = run(dataset, bits)

    out_json = os.path.join(HERE, "apples_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_json}")
