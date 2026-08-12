import os
from typing import BinaryIO
from collections import Counter

import regex as re

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_SPLIT_PATTERN_RE = re.compile(GPT2_SPLIT_PATTERN)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize(input_path, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")

    special_tokens = sorted(special_tokens, key=len, reverse=True)
    escaped_specials = [re.escape(token) for token in special_tokens]
    if escaped_specials:
        special_pattern = "|".join(escaped_specials)
        special_regex = re.compile(special_pattern)
    else:
        special_regex = None

    if special_regex:
        text_chunks = special_regex.split(text)
    else:
        text_chunks = [text]

    # Count all token spans in the text
    token_counter = Counter()

    for text_chunk in text_chunks:
        for token_match in GPT2_SPLIT_PATTERN_RE.finditer(text_chunk):
            token_str = token_match.group(0)
            token_bytes_tuple = tuple(bytes([b]) for b in token_str.encode("utf-8"))
            token_counter[token_bytes_tuple] += 1

    return dict(token_counter)


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):

    # vocabulary initialization
    vocab = {i: bytes([i]) for i in range(256)}
    if special_tokens:
        for token in special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")

    assert vocab_size >= len(vocab), "Vocab size must be greater than the number of special tokens"

    # pre-tokenize the text
    token_counter = pre_tokenize(input_path, special_tokens)

    def get_stats(bytes_tuple: tuple[bytes, ...], token_counter: dict = None, cnt: int = 0) -> Counter:
        counts = {} if token_counter is None else token_counter
        for pair in zip(bytes_tuple, bytes_tuple[1:]):
            counts[pair] = counts.get(pair, 0) + cnt
        return counts

    # merge tokens
    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        stats = {}
        for bytes_tuple, count in token_counter.items():
            get_stats(bytes_tuple, stats, count)

        # Prefer highest frequency; break ties by lexicographically largest pair (one pass).
        pair = max(stats, key=lambda p: (stats[p], p))

        # merge
        new_token = pair[0] + pair[1]
        vocab[len(vocab)] = new_token
        merges.append(pair)

        new_counter = {}
        for bytes_tuple, count in token_counter.items():
            new_byte_tuple = []
            i = 0
            while i < len(bytes_tuple):
                if i < len(bytes_tuple) - 1 and bytes_tuple[i] == pair[0] and bytes_tuple[i + 1] == pair[1]:
                    new_byte_tuple.append(new_token)
                    i += 2
                else:
                    new_byte_tuple.append(bytes_tuple[i])
                    i += 1

            new_counter[tuple(new_byte_tuple)] = new_counter.get(tuple(new_byte_tuple), 0) + count
        token_counter = new_counter

    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe("tests/fixtures/tinystories_sample.txt", 259, ["<|endoftext|>"])
    print(vocab)
    print(merges)
