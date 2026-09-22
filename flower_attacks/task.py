"""Model, data, and DLG attack for CIFAR-100 — exact port of the official notebook.

This file contains only ML logic (model, loss, attack). It has no Flower imports.
The Flower integration is in client_app.py and server_app.py.
"""

import os
import time
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torchvision import datasets, transforms


# Transforms
tp = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
])
tt = transforms.ToPILImage()


# Helpers
def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


# Weight init
def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


# LeNet (identical to notebook: 4 conv, 12 filters, Sigmoid)
class LeNet(nn.Module):
    """Shared collaborative model.

    num_classes is parameterised because the GAN attack of Hitaj et al.
    requires one extra output neuron, used by the adversary as the "fake"
    class into which GAN-generated samples are injected.
    """

    def __init__(self, num_classes: int = 100):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1), act(),
        )
        self.num_classes = num_classes
        self.fc = nn.Sequential(nn.Linear(768, num_classes))

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# --- Data ---
def load_cifar100():
    return datasets.CIFAR100("~/.torch", download=True)


# --- DLG attack from raw gradients (used by the server) ---
def extract_ground_truth_label(original_dy_dx):
    """Analytically extract the ground-truth label from shared gradients (iDLG).

    Implements Zhao et al., "iDLG: Improved Deep Leakage from Gradients" (2020), https://arxiv.org/abs/2001.02610 """
    last_layer_grad = original_dy_dx[-2]
    label = torch.argmin(torch.sum(last_layer_grad, dim=-1), dim=-1)
    return label.detach().reshape((1,)).requires_grad_(False)


def dlg_attack_from_gradients(
    net, original_dy_dx,
    input_shape=(1, 3, 32, 32), num_classes=100,
    num_iterations=300, save_dir="dlg_results", client_id=None,
    gt_image=None, use_idlg=False, stats_out=None, save_outputs=True,
    converge_threshold=1e-4,
):
    """DLG / iDLG attack given a model and intercepted gradients.

    Args:
        gt_image: optional PIL image of the ground truth (for plotting only,
                  NOT used in the attack itself).
        stats_out: optional dict, filled with timing and convergence data
                  (elapsed_s, ms_per_iter, converged_at, final_loss,
                  recovered_label) for benchmarking inside the federation.
        save_outputs: set False to skip writing images (benchmark runs).
        converge_threshold: gradient-matching loss counted as converged.
        use_idlg: if True, run iDLG (Zhao et al., 2020): the ground-truth
                  label is extracted analytically from the gradient signs and
                  held fixed, so only the dummy data is optimised. If False,
                  run the original DLG (Zhu et al., 2019), jointly optimising
                  the dummy data and a soft dummy label.
    """
    method = "iDLG" if use_idlg else "DLG"
    tag = f"client_{client_id}" if client_id is not None else "victim"
    save_dir = os.path.join(save_dir, tag)
    os.makedirs(save_dir, exist_ok=True)

    device = next(net.parameters()).device

    print(f"\n{'='*60}")
    print(f"  {method} Attack on {tag} — {num_iterations} L-BFGS iterations")
    print(f"{'='*60}")

    dummy_data = torch.randn(input_shape).to(device).requires_grad_(True)
    initial_noise = tt(dummy_data[0].cpu().detach())

    if use_idlg:
        # iDLG: extract the label analytically, then freeze it
        extracted_label = extract_ground_truth_label(original_dy_dx).to(device)
        print(f"  {tag} | Extracted label (analytic): {extracted_label.item()}")
        criterion = torch.nn.CrossEntropyLoss().to(device)
        # Only the dummy data is optimized
        optimizer = torch.optim.LBFGS([dummy_data])
        dummy_label = None
    else:
        # DLG: jointly optimise dummy data and a soft dummy label
        criterion = cross_entropy_for_onehot
        dummy_label = torch.randn((1, num_classes)).to(device).requires_grad_(True)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    history, losses = [], []
    t0 = time.perf_counter()
    converged_at = None

    for iters in range(num_iterations):
        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            if use_idlg:
                dummy_loss = criterion(pred, extracted_label)
            else:
                dummy_loss = criterion(pred, F.softmax(dummy_label, dim=-1))
            dummy_dy_dx = torch.autograd.grad(
                dummy_loss, net.parameters(), create_graph=True
            )
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        current_loss = closure()
        losses.append(current_loss.item())
        if converged_at is None and current_loss.item() < converge_threshold:
            converged_at = iters + 1
        if iters % 10 == 0:
            print(f"  {tag} | iter {iters:>3d} | loss {current_loss.item():.4f}")
        if save_outputs:
            history.append(tt(dummy_data[0].cpu()))

    elapsed = time.perf_counter() - t0

    if use_idlg:
        recovered_label = extracted_label.item()
    else:
        recovered_label = torch.argmax(dummy_label, dim=-1).item()
    print(f"  {tag} | Recovered label: {recovered_label} | "
          f"{elapsed:.2f}s ({1000 * elapsed / max(num_iterations, 1):.1f} ms/iter)")

    if stats_out is not None:
        stats_out.update({
            "method": method,
            "elapsed_s": elapsed,
            "ms_per_iter": 1000 * elapsed / max(num_iterations, 1),
            "converged_at": converged_at,
            "final_loss": losses[-1] if losses else float("nan"),
            "recovered_label": recovered_label,
            "iterations": num_iterations,
        })

    if save_outputs:
        save_results(history, losses, initial_noise, gt_image, save_dir,
                      f"{method} Recovery — {tag}")
        print(f"  Results saved to: {os.path.abspath(save_dir)}/")
    return dummy_data.detach(), (dummy_label.detach() if dummy_label is not None
                                 else extracted_label)


# DLG attack from dataset (standalone / run_dlg.py)
def dlg_attack(net, dst, img_index=25, num_iterations=300, save_dir="results", use_idlg=False):
    """DLG / iDLG attack from a dataset index.

    Args:
        use_idlg: if True, run iDLG (analytic label extraction, data-only
                  optimisation); if False, run the original DLG.
    """
    method = "iDLG" if use_idlg else "DLG"
    os.makedirs(save_dir, exist_ok=True)
    device = next(net.parameters()).device

    gt_data = tp(dst[img_index][0]).to(device).view(1, 3, 32, 32)
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device).view(1)
    gt_onehot_label = label_to_onehot(gt_label, num_classes=100)
    print("GT label is %d." % gt_label.item())

    # Victim's shared gradient
    out = net(gt_data)
    y = cross_entropy_for_onehot(out, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    initial_noise = tt(dummy_data[0].cpu().detach())

    if use_idlg:
        extracted_label = extract_ground_truth_label(original_dy_dx).to(device)
        correct = extracted_label.item() == gt_label.item()
        print("iDLG extracted label: %d (true %d) -> %s" % (
            extracted_label.item(), gt_label.item(),
            "CORRECT" if correct else "WRONG"))
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.LBFGS([dummy_data])
        dummy_label = None
    else:
        criterion = cross_entropy_for_onehot
        dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    history, losses = [], []
    t0 = time.perf_counter()
    converged_at = None

    for iters in range(num_iterations):
        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            if use_idlg:
                dummy_loss = criterion(pred, extracted_label)
            else:
                dummy_loss = criterion(pred, F.softmax(dummy_label, dim=-1))
            dummy_dy_dx = torch.autograd.grad(
                dummy_loss, net.parameters(), create_graph=True
            )
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        current_loss = closure()
        losses.append(current_loss.item())
        if iters % 10 == 0:
            print(iters, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))

    recovered = (extracted_label.item() if use_idlg
                 else torch.argmax(dummy_label, dim=-1).item())
    mse = torch.mean((dummy_data.detach() - gt_data) ** 2).item()
    print("Recovered label: %d | True label: %d | label %s" % (
        recovered, gt_label.item(),
        "OK" if recovered == gt_label.item() else "WRONG"))
    print("Reconstruction MSE: %.6f" % mse)

    save_results(history, losses, initial_noise, tt(gt_data[0].cpu()),
                  save_dir, f"{method} Recovery Progression (CIFAR-100)")
    print(f"\nResults saved to: {os.path.abspath(save_dir)}/")
    return dummy_data.detach(), (dummy_label.detach() if dummy_label is not None
                                 else extracted_label)


# Plotting
def save_results(history, losses, initial_noise, gt_image, save_dir, title):
    os.makedirs(save_dir, exist_ok=True)
    initial_noise.save(os.path.join(save_dir, "iter_0000_noise.png"))
    for i, img in enumerate(history):
        img.save(os.path.join(save_dir, f"iter_{i+1:04d}.png"))

    # 3×10 grid
    fig = plt.figure(figsize=(12, 8))
    for i in range(min(30, len(history) // 10)):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i * 10])
        plt.title("iter=%d" % (i * 10), fontsize=7)
        plt.axis("off")
    fig.savefig(os.path.join(save_dir, "grid_30.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Progression (4 columns)
    milestones = [m for m in [0, 1, 10, 20, 50, 100, 150, 200, 250] if m < len(history)]
    if history:
        milestones.append(len(history) - 1)
    all_imgs, all_titles = [], []
    if gt_image:
        all_imgs.append(gt_image); all_titles.append("Ground\ntruth")
    all_imgs.append(initial_noise); all_titles.append("Iter=0")
    for idx in milestones:
        all_imgs.append(history[idx]); all_titles.append("Iter=%d" % (idx + 1))
    n = len(all_imgs); cols = 4; rows = (n + cols - 1) // cols
    fig2, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    axes = axes.flatten()
    for i in range(len(axes)):
        if i < n:
            axes[i].imshow(all_imgs[i]); axes[i].set_title(all_titles[i], fontsize=9)
        axes[i].axis("off")
    fig2.suptitle(title, fontsize=13, y=1.01)
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, "progression.png"), bbox_inches="tight", dpi=150)
    plt.close(fig2)

    # Loss curve
    fig3, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(1, len(losses) + 1), losses)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Gradient matching loss")
    ax.set_title("DLG Convergence")
    if any(l > 0 for l in losses):
        ax.set_yscale("log")
    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, "loss_curve.png"), bbox_inches="tight", dpi=150)
    plt.close(fig3)
    if gt_image:
        gt_image.save(os.path.join(save_dir, "ground_truth.png"))
