"""Train the assignment BPE tokenizers and export reusable artifacts.

Writeup questions covered:
  - train_bpe_tinystories: maximum vocabulary size 10,000.
  - train_bpe_expts_owt: maximum vocabulary size 32,000, trained on the
    provided 5M-character OWT subset.

Artifacts per dataset go to experiments/artifacts/{name}/:
  vocab.json        {token_id: token_bytes as latin-1 string}
  merges.json       [[left, right], ...] as latin-1 strings, in merge order
  merge_counts.json corpus frequency of each merged pair (pre-merge)
  meta.json         run configuration and timing

Figures go to <repo_root>/notes/assignment1/figures/.

Usage:
  uv run python experiments/train_tokenizers.py
      [--dataset tinystories|owt|both] [--force]
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cs336_basics.tokenizer import train_bpe

ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
FIGURES_DIR = ASSIGNMENT_DIR.parent / "notes" / "assignment1" / "figures"

# bytes <-> str via latin-1 is lossless (every byte maps to a codepoint 0-255)
def _bytes_to_str(b: bytes) -> str:
    return b.decode("latin-1")


def _str_to_bytes(s: str) -> bytes:
    return s.encode("latin-1")


def save_tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({str(k): _bytes_to_str(v) for k, v in vocab.items()}, f, ensure_ascii=True)
    with open(out_dir / "merges.json", "w", encoding="utf-8") as f:
        json.dump([[_bytes_to_str(a), _bytes_to_str(b)] for a, b in merges], f, ensure_ascii=True)


def load_tokenizer_artifacts(out_dir: Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(out_dir / "vocab.json", encoding="utf-8") as f:
        vocab = {int(k): _str_to_bytes(v) for k, v in json.load(f).items()}
    with open(out_dir / "merges.json", encoding="utf-8") as f:
        merges = [(_str_to_bytes(a), _str_to_bytes(b)) for a, b in json.load(f)]
    return vocab, merges


def artifact_matches_config(out_dir: Path, vocab_size: int, special_tokens: list[str]) -> bool:
    """Return whether an existing artifact was trained with this configuration."""
    required_files = ("vocab.json", "merges.json", "merge_counts.json", "meta.json")
    if not all((out_dir / filename).exists() for filename in required_files):
        return False
    try:
        with open(out_dir / "meta.json", encoding="utf-8") as f:
            metadata = json.load(f)
        vocab, _ = load_tokenizer_artifacts(out_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        len(vocab) == vocab_size
        and metadata.get("vocab_size") == vocab_size
        and metadata.get("special_tokens") == special_tokens
    )


def plot_merge_counts(counts: list[int], name: str, n_base_tokens: int) -> None:
    """Log-log plot of the corpus frequency of the merged pair vs. merge step."""
    steps = list(range(1, len(counts) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.loglog(steps, counts)
    ax1.set_xlabel("merge step")
    ax1.set_ylabel("total count of merged pair")
    ax1.set_title(f"{name}: merge counts over training (log-log)")
    ax1.grid(True, which="both", alpha=0.3)

    vocab_sizes = [n_base_tokens + i for i in steps]
    ax2.loglog(vocab_sizes, counts)
    ax2.set_xlabel("vocabulary size")
    ax2.set_ylabel("total count of merged pair")
    ax2.set_title(f"{name}: merge counts vs. vocab size (log-log)")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"{name}_merge_counts_loglog.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved figure: {out_path}")


def run_dataset(name: str, config: dict, force: bool = False) -> None:
    corpus_path = config["corpus_path"]
    special_tokens = config["special_tokens"]
    vocab_size = config["vocab_size"]
    n_chars = config["n_chars"]
    artifact_name = config["artifact_name"]
    out_dir = ARTIFACTS_DIR / artifact_name

    print(f"\n=== {name} ===")
    print(f"corpus: {corpus_path} (first {n_chars:,} chars), vocab_size={vocab_size:,}, specials={special_tokens}")

    if not force and artifact_matches_config(out_dir, vocab_size, special_tokens):
        print(f"using existing artifact: {out_dir} (pass --force to retrain)")
        return

    t0 = time.time()
    vocab, merges, counts = train_bpe(str(corpus_path), vocab_size, special_tokens, return_history=True)
    elapsed = time.time() - t0
    print(f"training done in {elapsed / 60:.1f} min ({len(merges)} merges)")

    save_tokenizer(vocab, merges, out_dir)
    with open(out_dir / "merge_counts.json", "w") as f:
        json.dump(counts, f)
    longest_token_id, longest_token = max(vocab.items(), key=lambda item: (len(item[1]), item[0]))
    meta = {
        "dataset": name,
        "corpus": str(corpus_path),
        "corpus_chars": n_chars,
        "vocab_size": vocab_size,
        "actual_vocab_size": len(vocab),
        "special_tokens": special_tokens,
        "num_merges": len(merges),
        "train_seconds": round(elapsed, 1),
        "first_merge_count": counts[0] if counts else None,
        "last_merge_count": counts[-1] if counts else None,
        "longest_token_id": longest_token_id,
        "longest_token_bytes": list(longest_token),
        "longest_token_length": len(longest_token),
        "longest_token_utf8": longest_token.decode("utf-8", errors="replace"),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved artifacts: {out_dir}")
    if counts:
        print(f"first merge count = {counts[0]:,}, last merge count = {counts[-1]:,}")

    plot_merge_counts(counts, artifact_name, n_base_tokens=256 + len(special_tokens))


DATASETS = {
    "tinystories": {
        "corpus_path": ASSIGNMENT_DIR / "data" / "subsets" / "tinystories_50M.txt",
        "special_tokens": ["<|endoftext|>", "<|pad|>"],
        "vocab_size": 10_000,
        "artifact_name": "tinystories_10k",
        "n_chars": 50_000_000,
    },
    "owt": {
        "corpus_path": ASSIGNMENT_DIR / "data" / "subsets" / "owt_5M.txt",
        "special_tokens": ["<|endoftext|>"],
        "vocab_size": 32_000,
        "artifact_name": "owt_32k",
        "n_chars": 5_000_000,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*DATASETS, "both"], default="both")
    parser.add_argument("--force", action="store_true", help="retrain even when the artifact already exists")
    args = parser.parse_args()

    names = list(DATASETS) if args.dataset == "both" else [args.dataset]
    for name in names:
        cfg = DATASETS[name]
        assert cfg["corpus_path"].exists(), f"missing corpus: {cfg['corpus_path']}"
        run_dataset(name, cfg, force=args.force)


if __name__ == "__main__":
    main()
