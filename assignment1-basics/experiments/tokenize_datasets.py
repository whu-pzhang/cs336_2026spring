"""Encode train/valid corpora to uint16 .npy arrays for np.memmap loading.

Each file is encoded as one concatenated token-id sequence (documents are
already delimited by <|endoftext|> in the raw text). uint16 is enough for the
10K TinyStories and 32K OpenWebText vocabularies required by the assignment.

Writeup: tokenizer_experiments (d).

Usage:
  uv run python experiments/tokenize_datasets.py [--dataset tinystories|owt|both]
                                                 [--split train|valid|both]
                                                 [--workers N]
                                                 [--skip-existing]

Load later with:
  tokens = np.load(path, mmap_mode="r")
"""

import argparse
import io
import json
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer, find_chunk_boundaries

ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ASSIGNMENT_DIR / "data"
OUT_DIR = DATA_DIR / "tokenized"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

CHUNK_SIZE = 1_000_000
TARGET_CHUNK_BYTES = 128 * 1024 * 1024

JOBS = {
    "tinystories": {
        "tokenizer": "tinystories_10k",
        "special_tokens": ["<|endoftext|>", "<|pad|>"],
        "splits": {
            "train": DATA_DIR / "TinyStoriesV2-GPT4-train.txt",
            "valid": DATA_DIR / "TinyStoriesV2-GPT4-valid.txt",
        },
    },
    "owt": {
        "tokenizer": "owt_32k",
        "special_tokens": ["<|endoftext|>"],
        "splits": {
            "train": DATA_DIR / "owt_train.txt",
            "valid": DATA_DIR / "owt_valid.txt",
        },
    },
}


def load_tokenizer_artifacts(out_dir: Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Load the lossless latin-1 JSON format written by train_tokenizers.py."""
    with open(out_dir / "vocab.json", encoding="utf-8") as f:
        vocab = {int(k): v.encode("latin-1") for k, v in json.load(f).items()}
    with open(out_dir / "merges.json", encoding="utf-8") as f:
        merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in json.load(f)]
    return vocab, merges


def load_tokenizer(name: str) -> Tokenizer:
    vocab, merges = load_tokenizer_artifacts(ARTIFACTS_DIR / name)
    return Tokenizer(vocab, merges, special_tokens=JOBS[name]["special_tokens"])


def _encode_chunk(task: tuple[int, str, str, int, int, str]) -> dict:
    """Encode one byte-range and write bounded-size token chunks to disk."""
    chunk_index, src_name, temp_dir, start, end, tokenizer_name = task
    tokenizer = load_tokenizer(tokenizer_name)
    src = Path(src_name)
    temp_path = Path(temp_dir)

    with src.open("rb") as f:
        f.seek(start)
        raw = f.read(end - start)

    # TextIOWrapper gives encode_iterable a lazy line iterator without creating
    # a list of all lines. Boundaries are placed at special tokens, so each
    # chunk is an independent input to the tokenizer.
    token_buffer: list[int] = []
    part_paths: list[str] = []
    max_id = 0
    n_tokens = 0
    part_index = 0

    def flush_buffer() -> None:
        nonlocal part_index, n_tokens
        if not token_buffer:
            return
        part_path = temp_path / f"chunk_{chunk_index:05d}_{part_index:05d}.npy"
        np.save(part_path, np.asarray(token_buffer, dtype=np.uint16))
        part_paths.append(str(part_path))
        n_tokens += len(token_buffer)
        token_buffer.clear()
        part_index += 1

    with io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", errors="ignore") as text_stream:
        for token_id in tokenizer.encode_iterable(text_stream):
            if token_id > max_id:
                max_id = token_id
            token_buffer.append(token_id)
            if len(token_buffer) >= CHUNK_SIZE:
                flush_buffer()
    flush_buffer()

    if max_id > np.iinfo(np.uint16).max:
        raise ValueError(f"token id {max_id} does not fit in uint16")

    return {
        "chunk_index": chunk_index,
        "start": start,
        "end": end,
        "n_tokens": n_tokens,
        "max_id": max_id,
        "parts": part_paths,
    }


def encode_to_npy(
    tokenizer: Tokenizer,
    src: Path,
    dst: Path,
    workers: int = 1,
    tokenizer_name: str | None = None,
) -> dict:
    """Encode a corpus in parallel and assemble it directly into an `.npy` memmap."""
    n_bytes = src.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, workers)
    special_tokens = tokenizer.special_tokens
    if not special_tokens:
        raise ValueError("parallel dataset encoding requires at least one special token")

    if tokenizer_name is None:
        matches = [name for name, config in JOBS.items() if config["special_tokens"] == special_tokens]
        if len(matches) != 1:
            raise ValueError("tokenizer_name is required when special tokens do not identify the tokenizer")
        tokenizer_name = JOBS[matches[0]]["tokenizer"]

    # Use the longest configured special token as the document boundary.  The
    # worker chunks therefore never split an atomic special token.
    special_token = max(special_tokens, key=len)
    desired_chunks = max(workers, math.ceil(n_bytes / TARGET_CHUNK_BYTES))
    with src.open("rb") as f:
        boundaries = find_chunk_boundaries(f, desired_chunks, special_token.encode("utf-8"))

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix=f".{dst.stem}.chunks.", dir=dst.parent) as temp_dir:
        tasks = [
            (i, str(src), temp_dir, start, end, tokenizer_name)
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
            if end > start
        ]
        results: list[dict] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_encode_chunk, task) for task in tasks]
            pbar = tqdm(total=n_bytes, unit="B", unit_scale=True, desc=src.name)
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(result["end"] - result["start"])
            pbar.close()

        results.sort(key=lambda item: item["chunk_index"])
        n_tokens = sum(item["n_tokens"] for item in results)
        max_id = max((item["max_id"] for item in results), default=0)
        if max_id > np.iinfo(np.uint16).max:
            raise ValueError(f"token id {max_id} does not fit in uint16")

        # open_memmap writes the NumPy header once and then lets us fill the
        # final array in order without retaining all chunks in RAM.
        tokens = np.lib.format.open_memmap(dst, mode="w+", dtype=np.uint16, shape=(n_tokens,))
        offset = 0
        for result in results:
            for part_path in result["parts"]:
                part = np.load(part_path, mmap_mode="r")
                end = offset + part.size
                tokens[offset:end] = part
                offset = end
        tokens.flush()
        del tokens

    elapsed = time.time() - t0
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
        "workers": workers,
        "chunks": len(results),
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
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="number of tokenizer worker processes",
    )
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
            all_meta[key] = encode_to_npy(tokenizers[tok_name], src, dst, workers=args.workers, tokenizer_name=tok_name)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(all_meta, f, indent=2)

    print(f"\nsaved summary: {meta_path}")


if __name__ == "__main__":
    main()
