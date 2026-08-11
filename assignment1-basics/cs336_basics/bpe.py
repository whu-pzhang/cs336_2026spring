from collections import Counter

import regex as re

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


test_msg = """low low low low low lower lower widest widest widest newest newest newest newest newest newest"""

s = re.findall(GPT2_PAT, test_msg)


freq = Counter(test_msg.split())


class BPETokenizer:
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab = {i: bytes([i]) for i in range(256)}  # int -> bytes
        self.merges = []  # list[tuple[bytes, bytes]]
        self.special_tokens = special_tokens
        for token in special_tokens:
            self.vocab[len(self.vocab)] = token.encode("utf-8")

        assert vocab_size > len(self.vocab), (
            "vocab_size must be greater than the number of vocabulary tokens (including special tokens)"
        )
        self.num_merges = vocab_size - len(self.vocab)


def pre_tokenize(input_path, special_tokens: list[str]) -> list[str]:
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    for chunk in re.finditer(GPT2_PAT, text):
        pass
