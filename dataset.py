"""
Vocabulary and Dataset utilities for MS-COCO image captioning.
"""

import json
import os
import re
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ── Vocabulary ────────────────────────────────────────────────────────────────

class Vocabulary:
    """
    Simple word-level vocabulary built from COCO captions.
    
    Special tokens:
        <pad>  (0) — padding
        <sos>  (1) — start of sequence
        <eos>  (2) — end of sequence
        <unk>  (3) — unknown word
    """

    PAD, SOS, EOS, UNK = 0, 1, 2, 3
    SPECIAL = ["<pad>", "<sos>", "<eos>", "<unk>"]

    def __init__(self, min_freq=5):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}

    def build(self, captions):
        """Build vocab from a list of caption strings."""
        counter = Counter()
        for cap in captions:
            counter.update(self._tokenize(cap))

        self.word2idx = {tok: i for i, tok in enumerate(self.SPECIAL)}
        for word, freq in counter.most_common():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        print(f"Vocabulary built: {len(self.word2idx)} tokens (min_freq={self.min_freq})")

    def encode(self, caption, max_len=64):
        """String → padded token ids tensor."""
        tokens = [self.SOS] + [
            self.word2idx.get(w, self.UNK)
            for w in self._tokenize(caption)
        ][:max_len - 2] + [self.EOS]
        return tokens

    def decode(self, ids):
        """Token id list → caption string."""
        words = []
        for i in ids:
            w = self.idx2word.get(i, "<unk>")
            if w in ("<eos>", "<pad>"):
                break
            if w not in ("<sos>", "<unk>"):
                words.append(w)
        return " ".join(words)

    @staticmethod
    def _tokenize(text):
        return re.findall(r"\w+", text.lower())

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.word2idx, f)

    @classmethod
    def load(cls, path, min_freq=5):
        v = cls(min_freq)
        with open(path) as f:
            v.word2idx = json.load(f)
        v.idx2word = {int(v): k for k, v in v.word2idx.items()}
        return v

    def __len__(self):
        return len(self.word2idx)


# ── Dataset ───────────────────────────────────────────────────────────────────

def get_transform(train=True):
    """Standard ViT preprocessing."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class COCOCaptionDataset(Dataset):
    """
    MS-COCO caption dataset.
    
    Each __getitem__ returns ONE (image, caption) pair.
    Since each COCO image has 5 captions, the dataset has 5x more items
    than images — each caption gets its own training example.

    Args:
        image_dir:    path to train2017/ or val2017/ folder
        ann_file:     path to captions_train2017.json or captions_val2017.json
        vocab:        Vocabulary instance
        max_seq_len:  max caption length (tokens)
        train:        whether to apply training augmentations
        max_images:   optionally cap dataset size (for quick iteration)
    """

    def __init__(self, image_dir, ann_file, vocab, max_seq_len=64,
                 train=True, max_images=None):
        self.image_dir = image_dir
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.transform = get_transform(train)

        with open(ann_file) as f:
            data = json.load(f)

        # Build image_id → filename map
        id2file = {img["id"]: img["file_name"] for img in data["images"]}

        # Flatten: one (image_path, caption) per annotation
        self.samples = []
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in id2file:
                continue
            path = os.path.join(image_dir, id2file[img_id])
            self.samples.append((path, ann["caption"]))
            if max_images and len(self.samples) >= max_images * 5:
                break

        print(f"Dataset: {len(self.samples)} caption samples "
              f"({len(self.samples)//5} unique images)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        # Encode caption: [<sos>, w1, w2, ..., <eos>, <pad>, ...]
        token_ids = self.vocab.encode(caption, self.max_seq_len)

        # Pad to max_seq_len
        padded = token_ids + [self.vocab.PAD] * (self.max_seq_len - len(token_ids))
        padded = padded[:self.max_seq_len]

        tokens = torch.tensor(padded, dtype=torch.long)
        return pixel_values, tokens


def collate_fn(batch):
    """
    Collate a batch into (pixel_values, input_tokens, target_tokens, padding_mask).
    
    For teacher-forced training:
        input  = tokens[:-1]  (feed <sos>, w1, ..., wN-1 into decoder)
        target = tokens[1:]   (predict w1, ..., wN, <eos>)
    """
    pixel_values = torch.stack([b[0] for b in batch])
    tokens = torch.stack([b[1] for b in batch])   # (B, max_seq_len)

    input_tokens = tokens[:, :-1]   # (B, T-1) — decoder input
    target_tokens = tokens[:, 1:]   # (B, T-1) — what we predict

    # Padding mask: True where input is <pad>
    pad_mask = (input_tokens == 0)

    return pixel_values, input_tokens, target_tokens, pad_mask
