"""
RaBitQ-style scalar correction on top of turbovec's Lloyd-Max codebook.

Tests whether a per-vector correction scalar — computed at encode time
and applied at search time — recovers recall lost to systematic bias
in turbovec's inner-product estimates.

Pipeline (numpy reimplementation of turbovec/src/encode.rs):
    1. Normalize each data vector v to unit u = v / ||v||
    2. Rotate: u_rot = R @ u  (R = seeded random orthogonal)
    3. Quantize: each coord of u_rot -> nearest Lloyd-Max centroid (Beta dist)
    4. Reconstruct: x_hat = centroids[codes]
    5. Baseline score: ||v|| * <x_hat, y_rot>
    6. Corrected score: scalar_v * <x_hat, y_rot>
       where scalar_v is one of three forms tested.

Two correction forms tested:
    A. Regression-optimal (JL projection):
       scalar = ||v|| * cos(u_rot, x_hat) / ||x_hat||
    B. Paper formula (RaBitQ Section 2.2.3):
       scalar = ||v|| / (||x_hat|| * cos(u_rot, x_hat))

Runs across 3 datasets x 2 bit widths = 6 operating points, plots
recall@1@k.
"""

import json
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist

DATA_DIR = os.path.expanduser("~/data/py-turboquant")
RESULTS_DIR = os.path.dirname(__file__)
SEED = 42
N_DB = 100_000
K_VALUES = [1, 2, 4, 8, 16, 32, 64]


def lloyd_max_codebook(bits, dim, max_iter=200, tol=1e-12):
    """Match turbovec/src/codebook.rs — Lloyd-Max on Beta((d-1)/2, (d-1)/2)."""
    a = (dim - 1) / 2.0
    n_levels = 1 << bits
    std_dev = np.sqrt(2.0 * a / ((2.0 * a + 1.0) * 4.0 * a))
    spread = 3.0 * std_dev
    centroids = np.linspace(-spread, spread, n_levels, dtype=np.float64)

    for _ in range(max_iter):
        midpoints = (centroids[:-1] + centroids[1:]) / 2.0
        edges = np.concatenate(([-1.0], midpoints, [1.0]))

        u_lo = (edges[:-1] + 1.0) / 2.0
        u_hi = (edges[1:] + 1.0) / 2.0
        prob = beta_dist.cdf(u_hi, a, a) - beta_dist.cdf(u_lo, a, a)

        new_centroids = np.empty(n_levels, dtype=np.float64)
        for i in range(n_levels):
            if prob[i] < 1e-15:
                new_centroids[i] = centroids[i]
                continue
            xs = np.linspace(edges[i], edges[i + 1], 2049)
            pdf_xs = beta_dist.pdf((xs + 1) / 2, a, a) / 2
            new_centroids[i] = np.trapz(xs * pdf_xs, xs) / prob[i]

        if np.max(np.abs(centroids - new_centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return boundaries.astype(np.float32), centroids.astype(np.float32)


def random_rotation(dim, seed):
    """Deterministic random orthogonal via QR (matches turbovec/src/rotation.rs)."""
    rng = np.random.RandomState(seed)
    g = rng.standard_normal((dim, dim)).astype(np.float64)
    q, r = np.linalg.qr(g)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q * signs
    return q.astype(np.float32)


def encode(vectors, rotation, boundaries, centroids):
    norms = np.linalg.norm(vectors, axis=1).astype(np.float32)
    unit = vectors / np.clip(norms[:, None], 1e-10, None)
    rotated = unit @ rotation.T
    codes_idx = np.searchsorted(boundaries, rotated)
    x_hat = centroids[codes_idx]
    return norms, rotated, x_hat


def compute_corrections(rotated, x_hat, norms):
    inner = np.einsum("nd,nd->n", rotated, x_hat)
    xh_norm = np.linalg.norm(x_hat, axis=1)
    cos_uv = inner / np.clip(xh_norm, 1e-10, None)
    return {
        "baseline": norms,
        "form_A_regression": norms * cos_uv / np.clip(xh_norm, 1e-10, None),
        "form_B_paper": norms / np.clip(xh_norm * cos_uv, 1e-10, None),
    }, {
        "cos_mean": float(cos_uv.mean()),
        "cos_std": float(cos_uv.std()),
        "xh_norm_mean": float(xh_norm.mean()),
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
    "glove-200": load_glove,
    "openai-1536": lambda: load_openai(1536),
    "openai-3072": lambda: load_openai(3072),
}


def run(dataset, bit_width):
    print(f"\n=== {dataset}, {bit_width}-bit, seed={SEED} ===")
    t0 = time.time()
    database, queries, dim = DATASETS[dataset]()
    true_top1 = np.argmax(queries @ database.T, axis=1)
    print(f"  data + ground truth ({len(queries)} queries x {len(database)} db): {time.time() - t0:.1f}s")

    boundaries, centroids = lloyd_max_codebook(bit_width, dim)
    rotation = random_rotation(dim, SEED)

    t0 = time.time()
    norms, rotated, x_hat = encode(database, rotation, boundaries, centroids)
    print(f"  encode: {time.time() - t0:.1f}s")

    scalars, stats = compute_corrections(rotated, x_hat, norms)
    print(f"  cos(u, x_hat): mean={stats['cos_mean']:.4f} std={stats['cos_std']:.4f}; ||x_hat|| mean={stats['xh_norm_mean']:.4f}")

    queries_rot = (queries @ rotation.T).astype(np.float32)
    k_max = max(K_VALUES)
    recalls = {}
    for name, s in scalars.items():
        t0 = time.time()
        top = score_and_topk(queries_rot, x_hat, s, k_max)
        recalls[name] = {str(k): round(recall_at_1_at_k(true_top1, top, k), 4) for k in K_VALUES}
        print(f"  {name:<22} recall@1={recalls[name]['1']:.4f} ({time.time() - t0:.1f}s)")

    return {
        "dataset": dataset, "dim": dim, "bit_width": bit_width, "seed": SEED,
        "n_db": len(database), "n_queries": len(queries),
        "cos_stats": stats, "recall_at_1_at_k": recalls,
    }


def plot(results, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    datasets = ["glove-200", "openai-1536", "openai-3072"]
    bit_widths = [2, 4]
    series = [
        ("baseline", "baseline (||v|| only)", "C0", "-"),
        ("form_A_regression", "form A (cos / ||x_hat||)", "C1", "--"),
        ("form_B_paper", "form B (1 / (||x_hat||·cos))", "C2", "-"),
    ]
    for row, bits in enumerate(bit_widths):
        for col, ds in enumerate(datasets):
            ax = axes[row, col]
            key = f"{ds}_{bits}bit"
            if key not in results:
                ax.set_title(f"{ds} {bits}-bit (missing)")
                continue
            r = results[key]["recall_at_1_at_k"]
            for name, label, color, ls in series:
                ys = [r[name][str(k)] for k in K_VALUES]
                ax.plot(K_VALUES, ys, marker="o", label=label, color=color, linestyle=ls)
            ax.set_xscale("log", base=2)
            ax.set_xticks(K_VALUES)
            ax.set_xticklabels([str(k) for k in K_VALUES])
            ax.set_xlabel("k")
            ax.set_ylabel(f"recall@1@k ({bits}-bit)")
            ax.set_title(f"{ds} ({bits}-bit, d={results[key]['dim']})")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\nPlot saved to {out_path}")


def summary_table(results):
    """Print a table comparing recall@1 across all 6 cells."""
    print("\n" + "=" * 80)
    print(f"{'cell':<28} {'baseline':>10} {'form_A':>10} {'form_B':>10} {'Δ(B-base)':>10}")
    print("-" * 80)
    for key in sorted(results.keys()):
        r = results[key]["recall_at_1_at_k"]
        b = r["baseline"]["1"]
        a = r["form_A_regression"]["1"]
        bb = r["form_B_paper"]["1"]
        print(f"{key:<28} {b:>10.4f} {a:>10.4f} {bb:>10.4f} {bb - b:>+10.4f}")


if __name__ == "__main__":
    results = {}
    for dataset in ["glove-200", "openai-1536", "openai-3072"]:
        for bits in (2, 4):
            results[f"{dataset}_{bits}bit"] = run(dataset, bits)

    out_json = os.path.join(RESULTS_DIR, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_json}")

    plot(results, os.path.join(RESULTS_DIR, "recall_grid.png"))
    summary_table(results)
