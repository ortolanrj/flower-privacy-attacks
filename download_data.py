"""Pre-download the datasets. Run this once before `flwr run`.

Downloading from inside the Ray simulation is fragile: several ClientApp actors
may fetch into the same directory at once, which corrupts the archives and
surfaces as "RuntimeError: File not found or corrupted".

Usage:
    python download_data.py                 # MNIST (GAN) + CIFAR-100 (DLG)
    python download_data.py --only mnist
    MNIST_ROOT=/path/to/data python download_data.py
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["mnist", "cifar100", "all"], default="all")
    args = ap.parse_args()
    failed = []

    if args.only in ("mnist", "all"):
        print("=== MNIST (GAN attack) ===")
        try:
            from dlg_flower.gan_attack import prepare_mnist
            root = prepare_mnist()
            print(f"OK -> {root}\n")
        except Exception as e:
            print(f"FAILED:\n{e}\n")
            failed.append("mnist")

    if args.only in ("cifar100", "all"):
        print("=== CIFAR-100 (DLG / iDLG) ===")
        try:
            from dlg_flower.task import load_cifar100
            load_cifar100()
            print("OK\n")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}\n")
            failed.append("cifar100")

    if failed:
        print(f"Could not prepare: {', '.join(failed)}")
        sys.exit(1)
    print("All datasets ready. You can now run `flwr run ...`.")


if __name__ == "__main__":
    main()
