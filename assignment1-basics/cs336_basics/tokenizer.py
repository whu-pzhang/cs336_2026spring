import heapq
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from typing import BinaryIO

import regex as re

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# GPT-2 first splits text into manageable pieces, then applies BPE inside each piece.
# Keeping whitespace attached to the following word is part of the GPT-2 behavior.
GPT2_SPLIT_PATTERN_RE = re.compile(GPT2_SPLIT_PATTERN)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.

    Each requested boundary is moved forward to the next occurrence of the
    special token, so a special token is never split across two chunks.
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
    """Count GPT-2 pre-tokenized byte sequences in a training corpus.

    Special tokens are removed before applying the GPT-2 regex.  BPE training
    operates on byte symbols, so each regex match is converted to a tuple of
    one-byte elements and identical tuples are counted together.
    """
    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")

    # Match longer special tokens first (for example, a double end-of-text
    # token must not be consumed as two shorter tokens).
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


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], return_history: bool = False):
    """Train a byte-level BPE vocabulary by repeatedly merging the most common pair."""

    # vocabulary initialization
    vocab = {i: bytes([i]) for i in range(256)}
    if special_tokens:
        for token in special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")

    assert vocab_size >= len(vocab), "Vocab size must be greater than the number of special tokens"

    # pretok sequence → how many times it appears in the corpus
    word_counts = pre_tokenize(input_path, special_tokens)

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

    merges: list[tuple[bytes, bytes]] = []
    # corpus frequency of each chosen pair, recorded before the merge is applied
    merge_count_history: list[int] = []
    num_merges = vocab_size - len(vocab)

    # pair -> weighted occurrences across the corpus.  pair_to_words is the
    # important index: after a merge, only words containing that pair can change.
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    # count -> pairs currently having that count.  This avoids scanning every
    # pair on every merge while preserving the required lexicographic tie-break.
    count_buckets: dict[int, set[tuple[bytes, bytes]]] = defaultdict(set)
    active_counts: set[int] = set()
    count_heap: list[int] = []

    def update_pair_count(pair: tuple[bytes, bytes], delta: int) -> None:
        old_count = pair_counts.get(pair, 0)
        if old_count:
            count_buckets[old_count].discard(pair)
            if not count_buckets[old_count]:
                active_counts.discard(old_count)

        new_count = old_count + delta
        if new_count > 0:
            pair_counts[pair] = new_count
            count_buckets[new_count].add(pair)
            if new_count not in active_counts:
                active_counts.add(new_count)
                heapq.heappush(count_heap, -new_count)
        else:
            pair_counts.pop(pair, None)

    def word_pair_counts(word: tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
        return Counter(zip(word, word[1:]))

    def add_word(word: tuple[bytes, ...], freq: int) -> None:
        """Add a word type to pair counts and the reverse occurrence index."""
        for pair, occurrences in word_pair_counts(word).items():
            update_pair_count(pair, freq * occurrences)
            pair_to_words[pair].add(word)

    def remove_word(word: tuple[bytes, ...], freq: int) -> None:
        """Remove a word type from pair counts and the reverse occurrence index."""
        for pair, occurrences in word_pair_counts(word).items():
            update_pair_count(pair, -freq * occurrences)
            words = pair_to_words[pair]
            words.discard(word)
            if not words:
                del pair_to_words[pair]

    for word, freq in word_counts.items():
        add_word(word, freq)

    for _ in range(num_merges):
        while count_heap and -count_heap[0] not in active_counts:
            heapq.heappop(count_heap)
        if not count_heap:
            # Corpus is fully merged (every pre-token is a single symbol); stop early.
            print(f"train_bpe: no pairs left to merge after {len(merges)} merges; stopping early.")
            break

        best_count = -count_heap[0]
        # The heap selects the highest count; the set handles the required
        # lexicographically greatest tie-break exactly.
        best_pair = max(count_buckets[best_count])
        merges.append(best_pair)
        merge_count_history.append(best_count)

        merged = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged

        # Snapshot the set because removing and re-adding words mutates the index.
        affected_words = list(pair_to_words[best_pair])

        for word in affected_words:
            freq = word_counts.pop(word)
            remove_word(word, freq)
            new_word = merge_pair(word, best_pair)
            word_counts[new_word] = word_counts.get(new_word, 0) + freq
            add_word(new_word, freq)

    if return_history:
        return vocab, merges, merge_count_history
    return vocab, merges


class Tokenizer:
    """Encode and decode text using a byte-level vocabulary and BPE merge rules."""

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Keep both directions because encoding starts from bytes while
        # decoding starts from integer token IDs.
        self.id_to_bytes = {k: v for k, v in vocab.items()}
        self.bytes_to_id = {v: k for k, v in vocab.items()}

        # Every raw byte must exist in a byte-level vocabulary.
        self.byte_to_id = [self.bytes_to_id[bytes([i])] for i in range(256)]

        self.special_to_id = {}
        # Special tokens are atomic tokens and must bypass normal BPE merging.
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.bytes_to_id:
                new_id = max(self.vocab.keys()) + 1
                self.vocab[new_id] = tok_bytes
                self.bytes_to_id[tok_bytes] = new_id
                self.id_to_bytes[new_id] = tok_bytes
            self.special_to_id[tok] = self.bytes_to_id[tok_bytes]

        if self.special_tokens:
            # Capture the token as a group so split() keeps the special token
            # in the returned list instead of discarding it.
            pattern = "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
            self.special_splitter = re.compile(f"({pattern})")
        else:
            self.special_splitter = None

        # Convert byte-level merge rules to integer IDs once at initialization.
        # Mapping: (left_id, right_id) -> (merged_id, rank), where a lower rank
        # means that the pair must be merged earlier.
        self.merge_info: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, (b1, b2) in enumerate(merges):
            id1 = self.bytes_to_id[b1]
            id2 = self.bytes_to_id[b2]
            merged_bs = b1 + b2
            merged_id = self.bytes_to_id[merged_bs]
            self.merge_info[(id1, id2)] = (merged_id, rank)

        # Pretokenization repeats many common words.  Cache the final BPE result
        # to avoid rebuilding the merge structure for every occurrence.
        self.pretoken_cache: dict[bytes, tuple[int, ...]] = {}
        self.cache_capacity = 100_000

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """Load a JSON vocabulary and a two-column merge file from disk."""
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_json = json.load(f)
        vocab = {int(k): v.encode("utf-8") for k, v in vocab_json.items()}

        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=2)
                if len(parts) >= 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))
        return cls(vocab, merges, special_tokens)

    def _bpe_encode_pretoken(self, pre_token: bytes) -> list[int]:
        """Apply BPE to one regex pre-token represented as UTF-8 bytes.

        The heap always exposes the lowest-rank currently known pair.  The
        linked-list arrays make a merge local: only the pairs touching the new
        merged token need to be added back to the heap.
        """
        cached = self.pretoken_cache.get(pre_token)
        if cached is not None:
            # Dicts preserve insertion order; moving a hit to the end gives us
            # a small LRU cache when the capacity limit is reached.
            self.pretoken_cache.pop(pre_token)
            self.pretoken_cache[pre_token] = cached
            return list(cached)

        if not pre_token:
            return []

        # Start with one token for each byte.  A one-byte pre-token is already
        # maximally merged and is also a useful edge case to handle explicitly.
        token_ids = [self.byte_to_id[b] for b in pre_token]
        if len(token_ids) == 1:
            result = tuple(token_ids)
        else:
            # Store token IDs as a doubly linked list using array indices.  A
            # removed node remains in the arrays but is marked dead, which lets
            # us invalidate old heap entries without searching the heap.
            prev = [-1] + list(range(len(token_ids) - 1))
            next_ = list(range(1, len(token_ids))) + [-1]
            alive = [True] * len(token_ids)
            heap: list[tuple[int, int, int, int]] = []

            def add_pair(left: int) -> None:
                """Push a merge candidate for the adjacent live nodes."""
                right = next_[left]
                if right == -1:
                    return
                info = self.merge_info.get((token_ids[left], token_ids[right]))
                if info is not None:
                    _, rank = info
                    heapq.heappush(heap, (rank, left, token_ids[left], token_ids[right]))

            for index in range(len(token_ids) - 1):
                add_pair(index)

            while heap:
                rank, left, left_id, right_id = heapq.heappop(heap)
                right = next_[left]
                # Candidates become stale after neighboring merges.  Validate
                # both IDs and the current rank before applying the candidate.
                if (
                    not alive[left]
                    or right == -1
                    or not alive[right]
                    or token_ids[left] != left_id
                    or token_ids[right] != right_id
                    or self.merge_info.get((left_id, right_id), (None, None))[1] != rank
                ):
                    continue

                merged_id = self.merge_info[(left_id, right_id)][0]
                # Merge right into left, then reconnect left to the old right
                # neighbor.  The merge rank determines the exact BPE order.
                token_ids[left] = merged_id
                alive[right] = False
                next_[left] = next_[right]
                if next_[left] != -1:
                    prev[next_[left]] = left

                previous = prev[left]
                if previous != -1:
                    add_pair(previous)
                add_pair(left)

            # Walk from the head of the linked list and skip dead nodes.
            result_ids = []
            index = 0
            while index != -1:
                if alive[index]:
                    result_ids.append(token_ids[index])
                index = next_[index]
            result = tuple(result_ids)

        if self.cache_capacity > 0:
            # Evict the least-recently-used entry before inserting a new one.
            if len(self.pretoken_cache) >= self.cache_capacity:
                self.pretoken_cache.pop(next(iter(self.pretoken_cache)))
            self.pretoken_cache[pre_token] = result
        return list(result)

    def _encode_plain_text(self, text: str) -> list[int]:
        """Encode text that contains no special-token matches."""
        token_ids = []
        for match in GPT2_SPLIT_PATTERN_RE.finditer(text):
            pre_token = match.group().encode("utf-8")
            token_ids.extend(self._bpe_encode_pretoken(pre_token))
        return token_ids

    def encode(self, text: str) -> list[int]:
        """Encode text, preserving configured special tokens as atomic IDs."""
        if not self.special_tokens:
            return self._encode_plain_text(text)

        # The capturing group in special_splitter keeps special-token pieces in
        # `parts`; ordinary pieces continue through normal GPT-2 pre-tokenization.
        parts = self.special_splitter.split(text)
        result = []
        for part in parts:
            if part in self.special_to_id:
                result.append(self.special_to_id[part])
            else:
                result.extend(self._encode_plain_text(part))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode each string from an iterable without joining the input."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Concatenate token bytes and decode them as UTF-8."""
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
