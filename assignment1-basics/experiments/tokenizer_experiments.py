"""Run the tokenizer experiments from Assignment 1, Section 2.7.

The experiment samples ten documents from each corpus, evaluates both trained
tokenizers on both samples, reports bytes/token and throughput, and estimates
the time required to tokenize the 825 GB Pile dataset.

Usage:
  .venv/bin/python experiments/tokenizer_experiments.py [--seed 42]
"""

import argparse
import json
import mmap
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cs336_basics.tokenizer import Tokenizer

ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ASSIGNMENT_DIR / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
FIGURES_DIR = ASSIGNMENT_DIR.parent / "notes" / "assignment1" / "figures"

N_DOCUMENTS = 10
DEFAULT_SEED = 42
PILE_BYTES = 825_000_000_000

DATASETS = {
    "tinystories": {
        "corpus_path": DATA_DIR / "TinyStoriesV2-GPT4-train.txt",
        "artifact_dir": ARTIFACTS_DIR / "tinystories_10k",
        "special_tokens": ["<|endoftext|>", "<|pad|>"],
        "vocab_size": 10_000,
        "document_separator": b"<|endoftext|>",
    },
    "owt": {
        "corpus_path": DATA_DIR / "owt_train.txt",
        "artifact_dir": ARTIFACTS_DIR / "owt_32k",
        "special_tokens": ["<|endoftext|>"],
        "vocab_size": 32_000,
        # The provided owt-sample train file uses blank lines between records.
        "document_separator": b"\n\n",
    },
}


def load_tokenizer_artifacts(out_dir: Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Load the lossless latin-1 JSON format produced by train_tokenizers.py."""
    with open(out_dir / "vocab.json", encoding="utf-8") as f:
        vocab = {int(k): value.encode("latin-1") for k, value in json.load(f).items()}
    with open(out_dir / "merges.json", encoding="utf-8") as f:
        merges = [(left.encode("latin-1"), right.encode("latin-1")) for left, right in json.load(f)]
    return vocab, merges


def load_tokenizer(name: str) -> Tokenizer:
    config = DATASETS[name]
    artifact_dir = config["artifact_dir"]
    meta_path = artifact_dir / "meta.json"
    if not artifact_dir.exists():
        raise FileNotFoundError(
            f"missing {name} tokenizer artifact: {artifact_dir}; "
            f"train a {config['vocab_size']:,}-token vocabulary first"
        )

    vocab, merges = load_tokenizer_artifacts(artifact_dir)
    actual_size = len(vocab)
    expected_size = config["vocab_size"]
    if actual_size != expected_size:
        raise ValueError(
            f"{name} tokenizer has vocabulary size {actual_size:,}, "
            f"but Section 2.7 requires {expected_size:,}; artifact={artifact_dir}"
        )
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        if metadata.get("vocab_size") != expected_size:
            raise ValueError(f"{meta_path} reports an incompatible vocabulary size")
    return Tokenizer(vocab, merges, special_tokens=config["special_tokens"])


def sample_documents(path: Path, separator: bytes, count: int, seed: int) -> list[str]:
    """Randomly sample distinct non-empty documents using memory-mapped seeks.

    This avoids reading a multi-gigabyte corpus just to select ten documents.
    The separator is removed from the sampled document before tokenization.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    if not separator:
        raise ValueError("document separator cannot be empty")

    rng = random.Random(seed)
    file_size = path.stat().st_size
    samples: list[tuple[int, int, str]] = []
    spans: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(100, count * 100)

    with path.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        while len(samples) < count and attempts < max_attempts:
            attempts += 1
            position = rng.randrange(file_size)
            previous = mapped.rfind(separator, 0, position)
            start = 0 if previous < 0 else previous + len(separator)
            following = mapped.find(separator, start)
            end = file_size if following < 0 else following
            if end <= start:
                continue
            raw = bytes(mapped[start:end]).strip()
            if not raw or (start, end) in spans:
                continue
            spans.add((start, end))
            samples.append((start, end, raw.decode("utf-8", errors="ignore")))

    if len(samples) != count:
        raise RuntimeError(f"sampled only {len(samples)} non-empty documents from {path}")
    return [text for _, _, text in samples]


def encode_documents(tokenizer: Tokenizer, documents: list[str]) -> tuple[list[int], int, float]:
    """Encode sampled documents and return IDs, encoded bytes, and elapsed seconds."""
    token_ids: list[int] = []
    n_bytes = 0
    start = time.perf_counter()
    for document in documents:
        n_bytes += len(document.encode("utf-8"))
        token_ids.extend(tokenizer.encode(document))
    elapsed = time.perf_counter() - start
    return token_ids, n_bytes, elapsed


def plot_token_hist(ids: list[int], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(ids, bins=1000, log=True)
    ax.set_xlabel("token ID")
    ax.set_ylabel("count (log)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved figure: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Assignment 1 Section 2.7 tokenizer experiments")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--documents", type=int, default=N_DOCUMENTS)
    args = parser.parse_args()
    if args.documents <= 0:
        raise ValueError("--documents must be positive")

    tokenizers = {name: load_tokenizer(name) for name in DATASETS}
    documents = {
        name: sample_documents(config["corpus_path"], config["document_separator"], args.documents, args.seed + index)
        for index, (name, config) in enumerate(DATASETS.items())
    }

    results: dict[str, dict] = {}
    throughput_totals = {name: {"bytes": 0, "seconds": 0.0} for name in tokenizers}
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name, docs in documents.items():
        for tokenizer_name, tokenizer in tokenizers.items():
            ids, n_bytes, elapsed = encode_documents(tokenizer, docs)
            throughput = n_bytes / elapsed if elapsed else 0.0
            bytes_per_token = n_bytes / len(ids) if ids else 0.0
            pile_seconds = PILE_BYTES / throughput if throughput else None
            key = f"{dataset_name}_{tokenizer_name}"
            results[key] = {
                "dataset": dataset_name,
                "tokenizer": tokenizer_name,
                "documents": args.documents,
                "seed": args.seed,
                "n_tokens": len(ids),
                "n_bytes": n_bytes,
                "bytes_per_token": bytes_per_token,
                "tokens_per_byte": len(ids) / n_bytes if n_bytes else 0.0,
                "seconds": elapsed,
                "throughput_bytes_per_second": throughput,
                "pile_825GB_seconds": pile_seconds,
                "pile_825GB_hours": pile_seconds / 3600 if pile_seconds is not None else None,
            }
            throughput_totals[tokenizer_name]["bytes"] += n_bytes
            throughput_totals[tokenizer_name]["seconds"] += elapsed
            print(
                f"[{key}] documents={args.documents} tokens={len(ids):,} bytes={n_bytes:,} "
                f"bytes/token={bytes_per_token:.3f} throughput={throughput / 1e6:.2f} MB/s "
                f"Pile={pile_seconds / 3600:.2f} hours"
            )
            plot_token_hist(
                ids,
                f"{dataset_name} encoded with {tokenizer_name} tokenizer",
                FIGURES_DIR / f"{key}_token_id_hist.png",
            )

    for tokenizer_name, totals in throughput_totals.items():
        throughput = totals["bytes"] / totals["seconds"] if totals["seconds"] else 0.0
        results[f"throughput_{tokenizer_name}"] = {
            "tokenizer": tokenizer_name,
            "n_bytes": totals["bytes"],
            "seconds": totals["seconds"],
            "throughput_bytes_per_second": throughput,
            "pile_825GB_seconds": PILE_BYTES / throughput if throughput else None,
            "pile_825GB_hours": PILE_BYTES / throughput / 3600 if throughput else None,
        }

    out_path = ARTIFACTS_DIR / "tokenizer_experiments.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved summary: {out_path}")


if __name__ == "__main__":
    main()
