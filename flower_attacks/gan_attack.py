"""GAN reconstruction attack — Hitaj et al., "Deep Models Under the GAN" (CCS 2017).

https://arxiv.org/abs/1702.07464
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image


NUM_CLASSES = 11            # 10 real digits + 1 fake class for the attacker
FAKE_CLASS_INDEX = 10
VICTIM_CLASSES = (0, 1, 2, 3, 4)
ADVERSARY_CLASSES = (5, 6, 7, 8, 9)


# Collaboratively trained model (the discriminator, from the adversary's view)
class CNNClassifier(nn.Module):

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc0 = nn.Linear(64 * 7 * 7, 256)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        x = F.max_pool2d(torch.tanh(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc0(x))
        x = torch.tanh(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

# DCGAN generator
class Generator(nn.Module):
    """Latent vector -> 1x28x28 image in [-1, 1]."""

    def __init__(self, latent_dim: int = 100) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.project = nn.Linear(latent_dim, 256 * 7 * 7)
        self.main = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """N(0, 0.02^2) initialization, per Section 8.3 of the paper."""
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, mean=1.0, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z).view(-1, 256, 7, 7)
        return self.main(x)


# Data
MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),      # -> [-1, 1]
])


# MNIST extra mirrors
_MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]

_MANUAL_HINT = """
MNIST could not be downloaded automatically (all mirrors failed).

Fetch the four files manually and EXTRACT them into {root}/MNIST/raw/.
torchvision checks for the *decompressed* files, so the .gz must be unzipped:

    mkdir -p {root}/MNIST/raw && cd {root}/MNIST/raw
    BASE=https://storage.googleapis.com/cvdf-datasets/mnist
    for f in train-images-idx3-ubyte train-labels-idx1-ubyte \\
             t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
        wget $BASE/$f.gz && gunzip -f $f.gz
    done

The directory must then contain (no .gz extension):
    train-images-idx3-ubyte  train-labels-idx1-ubyte
    t10k-images-idx3-ubyte   t10k-labels-idx1-ubyte

Alternatively point MNIST_ROOT at a directory that already has MNIST/raw:
    MNIST_ROOT=/path/to/data flwr run . gan-simulation --run-config "use-gan=true"
"""


def mnist_root() -> str:
    """Absolute data root.

    Absolute matters: under the Ray simulation backend each ClientApp actor may
    have a different working directory, so a relative "./data" would resolve
    differently per actor (and trigger repeated downloads).
    """
    return os.path.abspath(os.environ.get("MNIST_ROOT", "./data"))


def _mnist_present(root: str, train: bool) -> bool:
    try:
        datasets.MNIST(root=root, train=train, download=False,
                       transform=MNIST_TRANSFORM)
        return True
    except Exception:
        return False


def prepare_mnist(root: str | None = None) -> str:
    """Download MNIST once, trying each mirror in turn.

    Call this from the ServerApp before training starts. Concurrent downloads by
    several Ray actors into the same directory are a known cause of corrupted
    archives, so clients should never be the ones to fetch the data.
    """
    root = root or mnist_root()
    os.makedirs(root, exist_ok=True)
    if _mnist_present(root, True) and _mnist_present(root, False):
        return root

    original = list(datasets.MNIST.mirrors)
    last_err = None
    for mirror in _MNIST_MIRRORS:
        datasets.MNIST.mirrors = [mirror]
        try:
            datasets.MNIST(root=root, train=True, download=True,
                           transform=MNIST_TRANSFORM)
            datasets.MNIST(root=root, train=False, download=True,
                           transform=MNIST_TRANSFORM)
            datasets.MNIST.mirrors = original
            print(f"MNIST ready at {root} (mirror: {mirror})")
            return root
        except Exception as e:                       # try the next mirror
            last_err = e
            print(f"  mirror failed: {mirror} ({type(e).__name__})")
            raw = os.path.join(root, "MNIST", "raw")
            if os.path.isdir(raw):
                for fn in os.listdir(raw):
                    if fn.endswith(".gz"):
                        try:
                            os.remove(os.path.join(raw, fn))
                        except OSError:
                            pass
    datasets.MNIST.mirrors = original
    raise RuntimeError(_MANUAL_HINT.format(root=root)) from last_err


def _load_full_mnist(train: bool = True) -> datasets.MNIST:
    """Load MNIST from disk."""
    root = mnist_root()
    try:
        return datasets.MNIST(root=root, train=train, download=False,
                              transform=MNIST_TRANSFORM)
    except Exception:
        prepare_mnist(root)
        return datasets.MNIST(root=root, train=train, download=False,
                              transform=MNIST_TRANSFORM)


def load_partition(is_victim: bool, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Disjoint class split: the adversary never owns the victim's classes."""
    trainset = _load_full_mnist(train=True)
    testset = _load_full_mnist(train=False)

    train_labels = np.asarray(trainset.targets)
    classes = VICTIM_CLASSES if is_victim else ADVERSARY_CLASSES
    idx = np.where(np.isin(train_labels, classes))[0]

    train_loader = DataLoader(Subset(trainset, idx.tolist()),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False)
    return train_loader, test_loader


# Local training of the shared model
def train_classifier(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    lr_decay: float,
    device: torch.device,
    injected_x: torch.Tensor | None = None,
    injected_y: torch.Tensor | None = None,
) -> float:
    """SGD on the shared model, optionally mixing in the mislabelled GAN samples.

    The injected samples are spread evenly across the batches rather than
    appended as one block, so that every SGD step feels the deception.
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0,
                                weight_decay=lr_decay)
    criterion = nn.NLLLoss()

    running, n_batches = 0.0, 0
    inj_n = int(injected_x.size(0)) if injected_x is not None else 0
    cursor = 0
    per_batch_inj = 0
    if inj_n > 0 and len(loader) > 0:
        per_batch_inj = max(1, inj_n // len(loader))

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if inj_n > 0:
                start = cursor % inj_n
                end = start + per_batch_inj
                if end <= inj_n:
                    ex_x = injected_x[start:end]
                    ex_y = injected_y[start:end]
                else:
                    ex_x = torch.cat([injected_x[start:], injected_x[: end - inj_n]])
                    ex_y = torch.cat([injected_y[start:], injected_y[: end - inj_n]])
                cursor = end % inj_n
                x = torch.cat([x, ex_x.to(device)], dim=0)
                y = torch.cat([y, ex_y.to(device)], dim=0)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1

    return running / max(n_batches, 1)


@torch.no_grad()
def evaluate_classifier(model: nn.Module, loader: DataLoader,
                        device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.NLLLoss(reduction="sum")
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        pred = logits[:, :10].argmax(dim=1)      # ignore the fake class
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def train_generator(
    generator: Generator,
    discriminator: nn.Module,
    target_class: int,
    latent_dim: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> float:
    """Train G so that the shared model attributes G(z) to `target_class`.

    D is held fixed: the adversary only queries the model it legitimately
    received. NLLLoss is used because the classifier emits log-probabilities.
    """
    discriminator.eval()
    for p in discriminator.parameters():
        p.requires_grad_(False)

    generator.train()
    optimizer = torch.optim.SGD(generator.parameters(), lr=lr, momentum=0.0)
    criterion = nn.NLLLoss()
    target = torch.full((batch_size,), target_class, dtype=torch.long, device=device)

    last_loss = 0.0
    for _ in range(steps):
        optimizer.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        fake = generator(z)
        loss = criterion(discriminator(fake), target)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

    for p in discriminator.parameters():
        p.requires_grad_(True)

    return last_loss


@torch.no_grad()
def generate_samples(generator: Generator, num_samples: int, latent_dim: int,
                     device: torch.device) -> torch.Tensor:
    generator.eval()
    z = torch.randn(num_samples, latent_dim, device=device)
    return generator(z)


# Visualization
def save_grid(images: torch.Tensor, path: str | os.PathLike, nrow: int = 8) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid = ((images.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    save_image(grid, str(path), nrow=nrow)


def save_comparison(target_class: int, generator: Generator, latent_dim: int,
                    device: torch.device, path: str | os.PathLike,
                    num_samples: int = 10) -> None:
    """Pair each reconstruction with its nearest real image of the target class
    under L1 distance, to visualise the fidelity of the leak."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mnist = _load_full_mnist(train=True)
    targets = torch.tensor(mnist.targets)
    class_idx = (targets == target_class).nonzero(as_tuple=True)[0]
    real_pool = torch.stack([mnist[int(i)][0] for i in class_idx])

    num_candidates = max(256, num_samples * 20)
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_candidates, latent_dim, device=device)
        all_fakes = generator(z).cpu()

    # pick a diverse subset of reconstructions (farthest-point sampling in L1)
    flat = all_fakes.view(num_candidates, -1)
    selected_idx = [0]
    for _ in range(num_samples - 1):
        dists = torch.cdist(flat, flat[selected_idx], p=1).min(dim=1).values
        dists[selected_idx] = -1
        selected_idx.append(int(dists.argmax()))

    fake_images = all_fakes[selected_idx]
    dists = torch.cdist(fake_images.view(num_samples, -1).float(),
                        real_pool.view(real_pool.size(0), -1).float(), p=1)
    matched_reals = real_pool[dists.argmin(dim=1)]

    matched_reals = ((matched_reals + 1.0) / 2.0).clamp(0.0, 1.0)
    fake_images = ((fake_images + 1.0) / 2.0).clamp(0.0, 1.0)

    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 5))
    axes[0, 0].set_ylabel("Closest\noriginal", fontsize=12, fontweight="bold",
                          rotation=0, labelpad=60, va="center")
    axes[1, 0].set_ylabel("GAN\nreconstruction", fontsize=12, fontweight="bold",
                          rotation=0, labelpad=60, va="center")
    for i in range(num_samples):
        axes[0, i].imshow(matched_reals[i].squeeze(0).numpy(), cmap="gray",
                          vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(fake_images[i].squeeze(0).numpy(), cmap="gray",
                          vmin=0, vmax=1)
        axes[1, i].axis("off")

    plt.tight_layout(rect=[0.08, 0, 1, 0.95])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
