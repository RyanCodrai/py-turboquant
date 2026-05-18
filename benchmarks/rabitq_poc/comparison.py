"""Build the final published-vs-POC-vs-Rust-prototype comparison table + plot."""

import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
PROTO_DIR = os.path.join(HERE, "rust_results")
POC_RESULTS = json.load(open(os.path.join(HERE, "results.json")))

CELLS = [
    ("glove_2bit", "glove-200_2bit", "GloVe-200 2-bit"),
    ("glove_4bit", "glove-200_4bit", "GloVe-200 4-bit"),
    ("d1536_2bit", "openai-1536_2bit", "OpenAI-1536 2-bit"),
    ("d1536_4bit", "openai-1536_4bit", "OpenAI-1536 4-bit"),
    ("d3072_2bit", "openai-3072_2bit", "OpenAI-3072 2-bit"),
    ("d3072_4bit", "openai-3072_4bit", "OpenAI-3072 4-bit"),
]
K_VALUES = [1, 2, 4, 8, 16, 32, 64]


def load_cell(fkey, pkey):
    base = json.load(open(os.path.join(RESULTS_DIR, f"recall_{fkey}.json")))
    proto = json.load(open(os.path.join(PROTO_DIR, f"recall_{fkey}.json")))
    poc = POC_RESULTS[pkey]["recall_at_1_at_k"]
    return base, proto, poc


def summary_table():
    print(f"{'cell':<22} {'old TV':>8} {'POC pred':>9} {'Rust proto':>11} {'Δ vs old':>10} {'FAISS':>8} {'beats FAISS':>12}")
    print("-" * 86)
    rows = []
    for fkey, pkey, label in CELLS:
        base, proto, poc = load_cell(fkey, pkey)
        rb = base["tq_recalls"]["1"]
        rp = proto["tq_recalls"]["1"]
        rf = base["faiss_recalls"]["1"]
        rp_poc = poc["form_B_paper"]["1"]
        beats = "YES" if rp > rf else ("tie" if rp == rf else "no")
        rows.append((label, rb, rp_poc, rp, rp - rb, rf, beats))
        print(f"{label:<22} {rb:>8.4f} {rp_poc:>9.4f} {rp:>11.4f} {rp-rb:>+10.4f} {rf:>8.4f} {beats:>12}")
    return rows


def plot():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    bit_widths = [2, 4]
    datasets = [
        ("glove", "GloVe-200"),
        ("d1536", "OpenAI-1536"),
        ("d3072", "OpenAI-3072"),
    ]
    pkey_map = {
        ("glove", 2): "glove-200_2bit",
        ("glove", 4): "glove-200_4bit",
        ("d1536", 2): "openai-1536_2bit",
        ("d1536", 4): "openai-1536_4bit",
        ("d3072", 2): "openai-3072_2bit",
        ("d3072", 4): "openai-3072_4bit",
    }

    for row, bits in enumerate(bit_widths):
        for col, (ds, label) in enumerate(datasets):
            ax = axes[row, col]
            fkey = f"{ds}_{bits}bit"
            pkey = pkey_map[(ds, bits)]
            base, proto, poc = load_cell(fkey, pkey)

            x = K_VALUES
            base_y = [base["tq_recalls"][str(k)] for k in x]
            proto_y = [proto["tq_recalls"][str(k)] for k in x]
            faiss_y = [base["faiss_recalls"][str(k)] for k in x]
            poc_y = [poc["form_B_paper"][str(k)] for k in x]

            ax.plot(x, base_y, marker="o", label="turbovec 0.4.3 (baseline)", color="C0", linewidth=2)
            ax.plot(x, proto_y, marker="s", label="prototype (Rust, corrected)", color="C3", linewidth=2)
            ax.plot(x, poc_y, marker="x", label="POC (numpy, corrected)", color="C2", linestyle="--", alpha=0.6)
            ax.plot(x, faiss_y, marker="^", label="FAISS PQ", color="C7", alpha=0.7)

            ax.set_xscale("log", base=2)
            ax.set_xticks(K_VALUES)
            ax.set_xticklabels([str(k) for k in K_VALUES])
            ax.set_xlabel("k")
            ax.set_ylabel(f"recall@1@k")
            ax.set_title(f"{label}, {bits}-bit")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "rust_comparison.png"), dpi=120)
    print(f"\nPlot saved to {os.path.join(HERE, 'rust_comparison.png')}")


if __name__ == "__main__":
    summary_table()
    plot()
