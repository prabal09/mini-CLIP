import csv
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from .tokenizer import SimpleWordTokenizer


class Flickr8kDataset(Dataset):
    """Flickr8k image-caption pairs.

    Expected directory layout (matches the Kaggle adityajn105/flickr8k mirror):
        root/
          Images/*.jpg
          captions.txt   (CSV with header "image,caption")

    Each (image, caption) row is one sample. With ~5 captions per image,
    there are ~40k pairs total over ~8k images.
    """

    def __init__(
        self,
        root: str | Path,
        tokenizer: SimpleWordTokenizer,
        max_seq_len: int,
        image_transform: Callable,
    ):
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.image_transform = image_transform
        self.samples: list[tuple[str, str]] = []
        with open(self.root / "captions.txt", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["image"], row["caption"]))

    @classmethod
    def all_captions(cls, root: str | Path) -> list[str]:
        """Read captions only — for tokenizer vocab building."""
        captions: list[str] = []
        with open(Path(root) / "captions.txt", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                captions.append(row["caption"])
        return captions

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_name, caption = self.samples[idx]
        image = Image.open(self.root / "Images" / img_name).convert("RGB")
        image = self.image_transform(image)
        tokens, eos_pos = self.tokenizer.encode(caption, self.max_seq_len)
        return image, torch.tensor(tokens, dtype=torch.long), eos_pos


def collate_fn(batch):
    images, tokens, eos_positions = zip(*batch)
    images = torch.stack(list(images), dim=0)
    tokens = torch.stack(list(tokens), dim=0)
    eos_positions = torch.tensor(eos_positions, dtype=torch.long)
    return images, tokens, eos_positions
