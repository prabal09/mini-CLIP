import re
from collections import Counter
from typing import Iterable


class SimpleWordTokenizer:
    """Word-level tokenizer built from a corpus of captions.

    Special tokens live at the lowest IDs:
        PAD = 0, BOS = 1, EOS = 2, UNK = 3
    Real words start at ID 4 and are ordered by descending frequency.

    Pedagogically clearer than BPE — every token is a human-readable word
    — at the cost of OOV handling: words below the frequency threshold,
    or unseen at test time, map to UNK. Acceptable for Flickr8k's small
    closed vocabulary.
    """

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3
    SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

    def __init__(self):
        self.word_to_id: dict[str, int] = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        self.id_to_word: list[str] = list(self.SPECIAL_TOKENS)

    @staticmethod
    def _normalize(text: str) -> list[str]:
        text = text.lower()
        # Replace any non-alphanumeric run with a space, then whitespace-split.
        text = re.sub(r"[^a-z0-9 ]", " ", text)
        return text.split()

    def build_vocab(self, captions: Iterable[str], min_freq: int = 3) -> None:
        counts: Counter[str] = Counter()
        for c in captions:
            counts.update(self._normalize(c))
        for word, freq in counts.most_common():
            if freq < min_freq:
                break
            if word not in self.word_to_id:
                self.word_to_id[word] = len(self.id_to_word)
                self.id_to_word.append(word)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_word)

    def encode(self, text: str, max_len: int) -> tuple[list[int], int]:
        """Returns (token_ids, eos_position).

        Layout: [BOS, w_1, …, w_k, EOS, PAD, …] padded to max_len.
        Captions longer than max_len - 2 are truncated to fit BOS + EOS.
        """
        words = self._normalize(text)[: max_len - 2]
        ids = [self.BOS] + [self.word_to_id.get(w, self.UNK) for w in words] + [self.EOS]
        eos_pos = len(ids) - 1
        ids = ids + [self.PAD] * (max_len - len(ids))
        return ids, eos_pos
