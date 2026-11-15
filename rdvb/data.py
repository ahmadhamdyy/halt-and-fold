"""ImageFolder loading without torchvision.

Train crops follow the usual ImageNet random-resized-crop settings.
Validation uses a 256 shorter-side resize and a 224 center crop.
"""

from __future__ import annotations

import math
import os
import random

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")


def list_classes(root: str) -> list[str]:
    names = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    return sorted(names)


class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool, class_to_idx: dict[str, int] | None = None, image_size: int = 224):
        self.root = root
        self.train = train
        self.image_size = image_size
        classes = list_classes(root)
        if class_to_idx is None:
            class_to_idx = {name: i for i, name in enumerate(classes)}
        self.class_to_idx = class_to_idx
        samples: list[tuple[str, int]] = []
        for name, idx in class_to_idx.items():
            folder = os.path.join(root, name)
            if not os.path.isdir(folder):
                continue
            for fn in sorted(os.listdir(folder)):
                if fn.endswith(_IMG_EXTS):
                    samples.append((os.path.join(folder, fn), idx))
        if not samples:
            raise FileNotFoundError(f"no images under {root}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = train_augment(img, self.image_size) if self.train else val_augment(img, self.image_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, label


def train_augment(img: Image.Image, size: int) -> Image.Image:
    img = random_resized_crop(img, size)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def val_augment(img: Image.Image, size: int) -> Image.Image:
    resize_to = int(round(size * 256 / 224))
    w, h = img.size
    scale = resize_to / min(w, h)
    nw, nh = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
    img = img.resize((nw, nh), Image.BICUBIC)
    left = max(0, (nw - size) // 2)
    top = max(0, (nh - size) // 2)
    return img.crop((left, top, left + size, top + size))


def random_resized_crop(img: Image.Image, size: int, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)) -> Image.Image:
    w, h = img.size
    area = w * h
    for _ in range(10):
        target = random.uniform(scale[0], scale[1]) * area
        aspect = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
        tw = int(round(math.sqrt(target * aspect)))
        th = int(round(math.sqrt(target / aspect)))
        if 0 < tw <= w and 0 < th <= h:
            left = random.randint(0, w - tw)
            top = random.randint(0, h - th)
            return img.crop((left, top, left + tw, top + th)).resize((size, size), Image.BICUBIC)
    return val_augment(img, size)


def load_imagenet_synset_index(path: str) -> dict[str, int]:
    import json

    with open(path) as f:
        raw = json.load(f)
    return {entry[0]: int(k) for k, entry in raw.items()}
