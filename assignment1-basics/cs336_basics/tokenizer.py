import os
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import BinaryIO, Optional
import json

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

    # pretok sequence (tuple of byte symbols) → corpus frequency
    word_counts = Counter()

    for text_chunk in text_chunks:
        for token_match in GPT2_SPLIT_PATTERN_RE.finditer(text_chunk):
            token_str = token_match.group(0)
            word = tuple(bytes([b]) for b in token_str.encode("utf-8"))
            word_counts[word] += 1

    return dict(word_counts)


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):

    # vocabulary initialization
    vocab = {i: bytes([i]) for i in range(256)}
    if special_tokens:
        for token in special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")

    assert vocab_size >= len(vocab), "Vocab size must be greater than the number of special tokens"

    # pretok sequence → how many times it appears in the corpus
    word_counts = pre_tokenize(input_path, special_tokens)

    def add_pair_counts(
        word: tuple[bytes, ...],
        pair_counts: dict | None = None,
        freq: int = 0,
    ) -> dict:
        """Add this word's adjacent pairs into pair_counts, weighted by freq."""
        counts = {} if pair_counts is None else pair_counts
        for pair in zip(word, word[1:]):
            counts[pair] = counts.get(pair, 0) + freq
        return counts

    def merge_pair(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def contain_pair(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> bool:
        for i in range(len(word) - 1):
            if word[i] == pair[0] and word[i + 1] == pair[1]:
                return True
        return False

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    # pair (left_symbol, right_symbol) → weighted occurrences across the corpus
    pair_counts = {}
    for word, freq in word_counts.items():
        add_pair_counts(word, pair_counts, freq)

    for _ in range(num_merges):
        # Prefer highest frequency; break ties by lexicographically largest pair (one pass).
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best_pair)

        merged = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged

        affected_words = []
        for word, freq in word_counts.items():
            if contain_pair(word, best_pair):
                affected_words.append((word, freq))

        for word, freq in affected_words:
            # remove best_pair from word
            add_pair_counts(word, pair_counts, -freq)
            new_word = merge_pair(word, best_pair)
            # add new pair counts
            add_pair_counts(new_word, pair_counts, freq)

            # update word counts
            word_counts[new_word] = word_counts.get(new_word, 0) + freq
            del word_counts[word]

        # remove 0 count pairs
        pair_counts = {p: c for p, c in pair_counts.items() if c > 0}

    return vocab, merges


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: Optional[list[str]] = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.id_to_bytes = {k: v for k, v in vocab.items()}
        self.bytes_to_id = {v: k for k, v in vocab.items()}

        # byte value -> token id
        self.byte_to_id = [self.bytes_to_id[bytes([i])] for i in range(256)]

        self.special_to_id = {}
        # add special tokens to vocab
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.bytes_to_id:
                new_id = max(self.vocab.keys()) + 1
                self.vocab[new_id] = tok_bytes
                self.bytes_to_id[tok_bytes] = new_id
                self.id_to_bytes[new_id] = tok_bytes
            self.special_to_id[tok] = self.bytes_to_id[tok_bytes]

        if self.special_tokens:
            pattern = "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
            self.special_splitter = re.compile(f"({pattern})")
        else:
            self.special_splitter = None

        # 构建 BPE 合并信息 (id pair -> (merged_id, rank))
        #   rank = 合并在 merges 列表中的位置（越小越先合并）
        self.merge_info: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, (b1, b2) in enumerate(merges):
            id1 = self.bytes_to_id[b1]
            id2 = self.bytes_to_id[b2]
            merged_bs = b1 + b2
            merged_id = self.bytes_to_id[merged_bs]
            self.merge_info[(id1, id2)] = (merged_id, rank)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[list[str]] = None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        vocab = {int(k): v.encode("utf-8") for k, v in vocab_json.items()}

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=2)
                if len(parts) >= 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))
        return cls(vocab, merges, special_tokens)

    def _bpe_encode_pretoken(self, pre_token: bytes) -> list[int]:
        parts = [self.byte_to_id[b] for b in pre_token]

        while True:
            best_rank = len(self.merges)
            best_idx = -1

            for i in range(len(parts) - 1):
                info = self.merge_info.get((parts[i], parts[i + 1]), None)
                if info is not None:
                    rank = info[1]
                    if rank < best_rank:
                        best_rank = rank
                        best_idx = i

            if best_idx == -1:
                break

            merged_id = self.merge_info[(parts[best_idx], parts[best_idx + 1])][0]
            parts[best_idx] = merged_id
            del parts[best_idx + 1]

        return parts

    def _encode_plain_text(self, text: str) -> list[int]:
        token_ids = []
        for match in GPT2_SPLIT_PATTERN_RE.finditer(text):
            pre_token = match.group().encode("utf-8")
            token_ids.extend(self._bpe_encode_pretoken(pre_token))
        return token_ids

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self._encode_plain_text(text)

        parts = self.special_splitter.split(text)
        result = []
        for part in parts:
            if part in self.special_to_id:
                result.append(self.special_to_id[part])
            else:
                result.extend(self._encode_plain_text(part))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        all_bytes = b"".join(self.id_to_bytes[id] for id in ids)
        return all_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
    from pathlib import Path
    from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path

    FIXTURES_PATH = Path("tests/fixtures")

    VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
    MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"])

    txt = "Hello <|endoftext|> world"

    tokenizer.encode(txt)
