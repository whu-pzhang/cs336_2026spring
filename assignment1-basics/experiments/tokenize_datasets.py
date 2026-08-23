"""Encode train/valid corpora to uint16 .npy arrays for np.memmap loading.

Each file is encoded as one concatenated token-id sequence (documents are
already delimited by <|endoftext|> in the raw text). uint16 is enough because
both tokenizers have vocab_size=32768.

Writeup: tokenizer_experiments (d).

Usage:
  uv run python experiments/tokenize_datasets.py [--dataset tinystories|owt|both]
                                                 [--split train|valid|both]
                                                 [--skip-existing]

Load later with:
  tokens = np.load(path, mmap_mode="r")
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
from train_tokenizers import ARTIFACTS_DIR, DATASETS, load_tokenizer_artifacts

from cs336_basics.tokenizer import Tokenizer

ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ASSIGNMENT_DIR / "data"
OUT_DIR = DATA_DIR / "tokenized"

CHUNK_SIZE = 1_000_000

JOBS = {
    "tinystories": {
        "tokenizer": "tinystories",
        "splits": {
            "train": DATA_DIR / "TinyStoriesV2-GPT4-train.txt",
            "valid": DATA_DIR / "TinyStoriesV2-GPT4-valid.txt",
        },
    },
    "owt": {
        "tokenizer": "owt",
        "splits": {
            "train": DATA_DIR / "owt_train.txt",
            "valid": DATA_DIR / "owt_valid.txt",
        },
    },
}


def load_tokenizer(name: str) -> Tokenizer:
    vocab, merges = load_tokenizer_artifacts(ARTIFACTS_DIR / name)
    return Tokenizer(vocab, merges, special_tokens=DATASETS[name]["special_tokens"])


def encode_to_npy(tokenizer: Tokenizer, src: Path, dst: Path) -> dict:
    n_bytes = src.stat().st_size
    chunks: list[np.ndarray] = []
    buf: list[int] = []
    max_id = 0
    n_tokens = 0

    t0 = time.time()
    pbar = tqdm(unit="tok", unit_scale=True, desc=src.name)
    with open(src, encoding="utf-8", errors="ignore") as f:
        for tok in tokenizer.encode_iterable(f):
            if tok > max_id:
                max_id = tok
            buf.append(tok)
            if len(buf) >= CHUNK_SIZE:
                chunks.append(np.asarray(buf, dtype=np.uint16))
                n_tokens += len(buf)
                pbar.update(len(buf))
                buf.clear()
        if buf:
            chunks.append(np.asarray(buf, dtype=np.uint16))
            n_tokens += len(buf)
            pbar.update(len(buf))
            buf.clear()
    pbar.close()

    if max_id > np.iinfo(np.uint16).max:
        raise ValueError(f"token id {max_id} does not fit in uint16")

    tokens = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint16)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, tokens)
    elapsed = time.time() - t0

    # Reload via memmap to confirm the training-time path works.
    loaded = np.load(dst, mmap_mode="r")
    assert loaded.dtype == np.uint16
    assert loaded.shape == (n_tokens,)

    meta = {
        "src": str(src),
        "dst": str(dst),
        "n_tokens": int(n_tokens),
        "n_bytes": int(n_bytes),
        "max_id": int(max_id),
        "dtype": "uint16",
        "bytes_per_token": round(n_bytes / n_tokens, 3) if n_tokens else None,
        "seconds": round(elapsed, 1),
        "throughput_MBps": round(n_bytes / elapsed / 1e6, 2) if elapsed else None,
    }
    print(
        f"  tokens={n_tokens:,}  max_id={max_id}  "
        f"bytes/token={meta['bytes_per_token']}  "
        f"{elapsed:.1f}s  ({meta['throughput_MBps']} MB/s)"
    )
    print(f"  saved {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*JOBS, "both"], default="both")
    parser.add_argument("--split", choices=["train", "valid", "both"], default="both")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    names = list(JOBS) if args.dataset == "both" else [args.dataset]
    splits = ["train", "valid"] if args.split == "both" else [args.split]

    tokenizers: dict[str, Tokenizer] = {}
    all_meta: dict[str, dict] = {}
    meta_path = OUT_DIR / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            all_meta = json.load(f)

    for name in names:
        job = JOBS[name]
        tok_name = job["tokenizer"]
        if tok_name not in tokenizers:
            print(f"loading tokenizer: {tok_name}")
            tokenizers[tok_name] = load_tokenizer(tok_name)

        for split in splits:
            src = job["splits"][split]
            dst = OUT_DIR / f"{name}_{split}.npy"
            key = f"{name}_{split}"
            assert src.exists(), f"missing corpus: {src}"
            if args.skip_existing and dst.exists():
                print(f"skip {key}: {dst} already exists")
                continue
            print(f"\n=== {key} ===")
            print(f"src: {src} ({src.stat().st_size / 1e9:.2f} GB)")
            all_meta[key] = encode_to_npy(tokenizers[tok_name], src, dst)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(all_meta, f, indent=2)

    print(f"\nsaved summary: {meta_path}")


if __name__ == "__main__":
    main()
