# Privacy Attacks & Defences in Federated Learning — Flower + PyTorch

A single Flower project implementing three privacy attacks on federated
learning and one cryptographic defence, all selectable by configuration flags.

| Mode | Flag | Attack | Adversary | Dataset |
|------|------|--------|-----------|---------|
| DLG  | *(default)* | Deep Leakage from Gradients (Zhu et al., NeurIPS 2019) | server | CIFAR-100 |
| iDLG | `use-idlg=true` | Improved DLG (Zhao et al., 2020) | server | CIFAR-100 |
| PHE  | `use-encryption=true` | *Additively Homomorphic Encryption* (Phong et al., 2017) | — | CIFAR-100 |
| GAN  | `use-gan=true` | Deep Models Under the GAN (Hitaj et al., CCS 2017) | **client** | MNIST |

PHE composes with every mode. Against DLG/iDLG it is a complete defence;
against the GAN attack it provably changes nothing — see below.

## Install

```bash
cd dlg_flower
pip install -e .
```

## Prepare the datasets first

```bash
python download_data.py            # MNIST + CIFAR-100
python download_data.py --only mnist
```

Run this **before** `flwr run`. Downloading from inside the simulation is
fragile: several Ray ClientApp actors may fetch into the same directory at
once, which corrupts the archives and surfaces as
`RuntimeError: File not found or corrupted`. The ServerApp also calls
`prepare_mnist()` once at startup as a safety net.

Point `MNIST_ROOT` at an existing copy to skip downloading entirely:

```bash
MNIST_ROOT=/path/to/data flwr run . gan-simulation --run-config "use-gan=true"
```

## Quick start

```bash
# 1. DLG attack (malicious server inverts a client's gradient)
flwr run . --run-config "use-idlg=false" --stream

# 2. iDLG (analytic label extraction, 100% label accuracy)
flwr run . --run-config "use-idlg=true" --stream

# 3. PHE defence — gradients encrypted, server cannot invert them
flwr run . --run-config "use-idlg=true use-encryption=true" --stream

# 4. GAN reconstruction attack (malicious client, honest server)
flwr run . gan-simulation --run-config "use-gan=true num-server-rounds=160" --stream

# 5. GAN attack WITH PHE — updates encrypted, attack unaffected
flwr run . gan-simulation --run-config \
  "use-gan=true use-encryption=true num-server-rounds=160 max-encrypt-values=500" --stream

# Standalone helpers (no Flower runtime)
python run_dlg.py --index 25 --iters 300 --compare   # DLG vs iDLG
python benchmark_dlg_idlg.py --trials 5 --iters 150  # timing comparison
python demo_defense.py --key-length 1024 --iters 300 # PHE defeats DLG
python demo_gan_phe.py --rounds 4                    # PHE does NOT defeat GAN
```

## Layout

```
dlg_flower/
├── dlg_flower/
│   ├── task.py         # LeNet, CIFAR-100, DLG + iDLG attacks, label extraction
│   ├── crypto.py       # PaillierCrypto: keygen, encrypt, homomorphic add, decrypt
│   ├── gan_attack.py   # CNNClassifier, Generator, MNIST split, GAN attack
│   ├── client_app.py   # @app.train() / @app.evaluate()
│   └── server_app.py   # @app.main(); MaliciousFedAvg + HonestFedAvg
├── run_dlg.py          # standalone DLG/iDLG runner
├── download_data.py    # one-time dataset download (run before flwr run)
└── pyproject.toml
```

---

## Attack variants: DLG and iDLG

The codebase implements both gradient-inversion attacks, selected by a flag:

| Variant | Reference | Label handling | Optimised variables |
|---------|-----------|----------------|---------------------|
| **DLG**  | Zhu et al., NeurIPS 2019 | soft label optimised jointly | `dummy_data`, `dummy_label` |
| **iDLG** | Zhao et al., 2020 ([arXiv:2001.02610](https://arxiv.org/abs/2001.02610)) | extracted analytically from gradient signs, then **frozen** | `dummy_data` only |

iDLG exploits the fact that, for cross-entropy over one-hot labels, the
gradient w.r.t. the correct logit lies in `(-1, 0)` while all others lie in
`(0, 1)`. Since `grad(W_L^i) = g_i * a_{L-1}` and `a_{L-1}` is independent of
`i`, the ground-truth label is the unique index whose last-layer weight
gradient has the opposite sign (Eq. 4 of the paper). This is implemented in
`extract_ground_truth_label()` and yields **100% label accuracy**.

### Running

```bash
flwr run . --run-config "use-idlg=true"                       # iDLG
flwr run . --run-config "use-idlg=true use-encryption=true"   # iDLG + PHE defence
```

## Attack variant: GAN reconstruction (Hitaj et al., CCS 2017)

Enabled with `use-gan`.

### Different threat model — the adversary is a CLIENT

| | DLG / iDLG | GAN attack |
|---|---|---|
| Attack lives in | `server_app.py` (`MaliciousFedAvg`) | `client_app.py` (`@app.train()`) |
| Server behaviour | malicious | **honest** (`HonestFedAvg`) |
| Exploited asset | individual gradient | shared global model |
| Dataset | CIFAR-100 | **MNIST** |
| Model | `LeNet` (sigmoid) | `CNNClassifier` (tanh, log_softmax) |
| Rounds needed | 1 | ~160 |
| Defeated by PHE | yes | **no** |

### Setup

* 11 classes: 10 digits + one **fake** class (index 10) used by the attacker.
* Disjoint split: **victim = partition 0** (digits 0–4),
  **adversary = partition 1** (digits 5–9).
* `target-class = 4` belongs to the victim and is never owned by the adversary.

### Mechanism (Algorithm 1)

Each round the adversary: loads the downloaded model as discriminator `D`;
trains its generator `G` (SGD, lr 0.02, 100 steps) so `D` labels `G(z)` as the
target class; samples 64 images and labels them with the **fake** class; and
merges them into its local SGD. That last step is the deception — and it is
also what supplies the adversarial pressure that keeps `G` realistic.

Generator state persists across rounds via `context.state`, so the
reconstruction improves cumulatively.

### Running

```bash
# faithful 2-participant setup (victim + adversary)
flwr run . gan-simulation --run-config "use-gan=true num-server-rounds=160" --stream
```

Outputs land in `outputs/reconstructions/round_XXXX.png` every `save-every`
rounds, plus `comparacao_original_vs_reconstrucao.png` on the final round,
pairing each reconstruction with its nearest real image under L1 distance.

### Configuration

| Key | Default | Meaning |
|-----|---------|---------|
| `use-gan` | `false` | enable the GAN reconstruction attack |
| `target-class` | `4` | victim's class, never owned by the adversary |
| `gan-adversary-id` | `1` | `partition_id` of the malicious client |
| `gan-latent-dim` | `100` | generator latent dimension |
| `gan-steps-per-round` | `100` | generator updates per FL round |
| `gan-batch-size` | `64` | generator batch size |
| `gan-lr` | `0.02` | SGD lr for `G` (Section 8.3 of the paper) |
| `num-injected-samples` | `64` | mislabelled samples injected per round |
| `batch-size` / `local-epochs` | `64` / `1` | local training of the shared model |
| `learning-rate` / `lr-decay` | `0.001` / `1e-7` | local SGD for the shared model |
| `save-every` | `5` | rounds between saved reconstruction grids |

The flags compose: `use-gan` selects the client-side attack with an honest
server, while `use-idlg` / `use-encryption` govern the server-side attack and
its cryptographic defence.

## Running the GAN attack with PHE

`use-gan` and `use-encryption` compose. With both enabled, each client encrypts
its model update under Paillier and `HonestFedAvg` aggregates the **ciphertexts**
homomorphically, so the server never observes an individual plaintext update:

```bash
flwr run . gan-simulation --run-config \
  "use-gan=true use-encryption=true num-server-rounds=160 max-encrypt-values=500" --stream
```

`max-encrypt-values` caps how many update components are actually encrypted, so
the demonstration stays tractable; the argument extends to the full update by
IND-CPA.

### Benchmark DLG and iDLG

Set `benchmark-attacks=true`. Each round the malicious server intercepts a real
victim gradient and runs **both** DLG and iDLG on that *same* gradient, so the
comparison is paired and uses genuine federated updates:

```bash
flwr run . --run-config "benchmark-attacks=true dlg-iterations=150" --stream
```

