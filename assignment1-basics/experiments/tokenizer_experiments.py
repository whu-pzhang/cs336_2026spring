"""Tokenizer experiments (writeup section 3.3).

Uses the two trained tokenizers (experiments/artifacts/{tinystories,owt}) to
encode both corpora, then:
  - reports the ratio of corpus size in tokens to corpus size in bytes,
  - plots histograms of token IDs (log y-axis), one figure per
    (dataset, tokenizer) pair.

Corpora: the same subsets the tokenizers were trained on
(data/subsets/tinystories_50M.txt, data/subsets/owt_5M.txt), so both are
encoded with both tokenizers.

Usage:
  uv run python experiments/tokenizer_experiments.py
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from train_tokenizers import ARTIFACTS_DIR, DATASETS, FIGURES_DIR, load_tokenizer_artifacts

from cs336_basics.tokenizer import Tokenizer


def load_tokenizer(name: str) -> Tokenizer:
    vocab, merges = load_tokenizer_artifacts(ARTIFACTS_DIR / name)
    return Tokenizer(vocab, merges, special_tokens=DATASETS[name]["special_tokens"])


def encode_file(tokenizer: Tokenizer, path: Path) -> tuple[list[int], int]:
    """Encode the whole file; return (token_ids, number of bytes actually encoded)."""
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="ignore")
    ids = tokenizer.encode(text)
    return ids, len(text.encode("utf-8"))


def plot_token_hist(ids: list[int], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        ax.hist(ids, bins=1000, log=True)
    except Exception:
        # older matplotlib versions mis-handle log=True with empty bins
        counts, edges = np.histogram(ids, bins=1000)
        ax.bar(edges[:-1], counts, width=(edges[1] - edges[0]), align="edge", log=True)
    ax.set_xlabel("token ID")
    ax.set_ylabel("count (log)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved figure: {out_path}")


def main():
    tokenizers = {name: load_tokenizer(name) for name in DATASETS}
    corpora = {name: cfg["corpus_path"] for name, cfg in DATASETS.items()}

    results = {}
    for ds_name, corpus_path in corpora.items():
        for tok_name, tok in tokenizers.items():
            t0 = time.time()
            ids, n_bytes = encode_file(tok, corpus_path)
            elapsed = time.time() - t0
            n_tokens = len(ids)
            ratio = n_tokens / n_bytes
            print(
                f"[{ds_name} x {tok_name}-tokenizer] tokens={n_tokens:,} bytes={n_bytes:,} "
                f"ratio(tokens/bytes)={ratio:.4f}  ({1 / ratio:.2f} bytes/token)  [{elapsed:.1f}s]"
            )
            results[f"{ds_name}_{tok_name}"] = {
                "dataset": ds_name,
                "tokenizer": tok_name,
                "n_tokens": n_tokens,
                "n_bytes": n_bytes,
                "token_to_byte_ratio": ratio,
            }
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            plot_token_hist(
                ids,
                f"{ds_name} encoded with {tok_name} tokenizer",
                FIGURES_DIR / f"{ds_name}_{tok_name}_token_id_hist.png",
            )

    out_path = ARTIFACTS_DIR / "tokenizer_experiments.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved summary: {out_path}")


if __name__ == "__main__":
    main()
