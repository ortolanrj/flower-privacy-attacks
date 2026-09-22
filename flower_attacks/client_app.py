"""ClientApp — honest participants, optional Paillier encryption, optional GAN adversary.

Three modes, selected from the train ConfigRecord:

  * default (DLG / iDLG baseline) — CIFAR-100 + LeNet. Every client returns the
    gradient of a single private sample; the SERVER is the adversary.

  * use-encryption — the client encrypts that gradient under the shared Paillier
    public key, so the server holds only E(G) and cannot invert it.

  * use-gan — Hitaj et al., "Deep Models Under the GAN" (CCS 2017), on MNIST.
    Here the adversary is a CLIENT: partition 0 is the honest victim (owning
    digits 0-4) and partition 1 is the attacker (owning digits 5-9). The
    attacker trains a local generator against the shared model and injects
    mislabelled samples. The server stays honest in this mode.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flower_attacks.crypto import PaillierCrypto
from flower_attacks.gan_attack import (
    FAKE_CLASS_INDEX,
    CNNClassifier,
    Generator,
    evaluate_classifier,
    generate_samples,
    load_partition,
    save_comparison,
    save_grid,
    train_classifier,
    train_generator,
)
from flower_attacks.task import (
    LeNet,
    cross_entropy_for_onehot,
    label_to_onehot,
    load_cifar100,
    tp,
)

app = ClientApp()

_GEN_KEY = "adv_generator"
_CFG_KEY = "adv_state"


# Adversary generator persistence
def is_gan_state(state_dict) -> bool:
    """Detect which architecture a received state_dict belongs to.

    CNNClassifier (GAN scenario) has conv1/conv2/fc0..fc2; LeNet (DLG/iDLG) has
    body.*/fc.*. Inferring from the payload rather than trusting the config
    alone makes the client robust to a missing flag, which would otherwise
    surface as an opaque "Missing key(s) in state_dict" error.
    """
    return any(k.startswith("conv1.") or k.startswith("fc0.") for k in state_dict)


def load_or_init_generator(context: Context, latent_dim: int, device):
    generator = Generator(latent_dim=latent_dim).to(device)
    if _GEN_KEY in context.state:
        generator.load_state_dict(
            context.state[_GEN_KEY].to_torch_state_dict(), strict=True
        )
    round_counter = 0
    if _CFG_KEY in context.state:
        round_counter = int(context.state[_CFG_KEY]["round_counter"])
    return generator, round_counter


def save_generator(context: Context, generator: Generator, round_counter: int):
    context.state[_GEN_KEY] = ArrayRecord(generator.state_dict())
    context.state[_CFG_KEY] = ConfigRecord({"round_counter": round_counter})


# GAN scenario (Hitaj et al., Algorithm 1)
def gan_train(msg: Message, context: Context, cfg: ConfigRecord) -> Message:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    partition_id = int(context.node_config["partition-id"])
    is_adversary = (partition_id == int(cfg.get("gan-adversary-id", 1)))

    batch_size = int(cfg.get("batch-size", 64))
    local_epochs = int(cfg.get("local-epochs", 1))
    lr = float(cfg.get("learning-rate", 0.001))
    lr_decay = float(cfg.get("lr-decay", 1e-7))

    # use the data root the server already prepared (absolute path)
    if "mnist-root" in cfg:
        os.environ.setdefault("MNIST_ROOT", str(cfg["mnist-root"]))

    model = CNNClassifier().to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    train_loader, _ = load_partition(is_victim=not is_adversary,
                                     batch_size=batch_size)

    # Flower's FedAvg requires every MetricRecord in a round to carry the SAME
    # keys, so victim and adversary must report an identical schema. Fields
    # that only make sense for the attacker are reported as 0 by the victim.
    metrics = {
        "partition_id": float(partition_id),
        "role": 1.0 if is_adversary else 0.0,
        "train_loss": 0.0,
        "num-examples": 0.0,
        "gan_g_loss": 0.0,
        "round_counter": 0.0,
        "encrypted": 0.0,
        "n_encrypted": 0.0,
    }

    if not is_adversary:
        loss = train_classifier(model, train_loader, local_epochs, lr, lr_decay, device)
        n_examples = len(train_loader.dataset)
    else:
        latent_dim = int(cfg.get("gan-latent-dim", 100))
        target_class = int(cfg.get("target-class", 4))
        n_inject = int(cfg.get("num-injected-samples", 64))
        save_every = int(cfg.get("save-every", 5))
        num_rounds = int(cfg.get("num-server-rounds", 160))
        output_dir = str(cfg.get("output-dir", "outputs"))

        generator, round_counter = load_or_init_generator(context, latent_dim, device)
        round_counter += 1

        # Train G against the freshly downloaded shared model
        g_loss = train_generator(
            generator=generator, discriminator=model,
            target_class=target_class, latent_dim=latent_dim,
            steps=int(cfg.get("gan-steps-per-round", 100)),
            batch_size=int(cfg.get("gan-batch-size", 64)),
            lr=float(cfg.get("gan-lr", 0.02)), device=device,
        )

        # Mislabel the generated samples as the FAKE class
        injected_x = generate_samples(generator, n_inject, latent_dim,
                                      device).detach().cpu()
        injected_y = torch.full((n_inject,), FAKE_CLASS_INDEX, dtype=torch.long)

        # Local SGD on real data plus the deceptive samples
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        loss = train_classifier(model, train_loader, local_epochs, lr,
                                lr_decay, device,
                                injected_x=injected_x, injected_y=injected_y)
        n_examples = len(train_loader.dataset) + n_inject

        if round_counter == 1 or round_counter % save_every == 0:
            out = Path(output_dir) / "reconstructions" / f"round_{round_counter:04d}.png"
            save_grid(generate_samples(generator, 64, latent_dim, device), out, nrow=8)
            print(f"  [adversary] reconstruction grid -> {out}")

        if round_counter == num_rounds:
            cmp_path = Path(output_dir) / "comparacao_original_vs_reconstrucao.png"
            save_comparison(target_class, generator, latent_dim, device,
                            cmp_path, num_samples=10)
            print(f"  [adversary] final comparison -> {cmp_path}")

        save_generator(context, generator, round_counter)
        metrics["gan_g_loss"] = float(g_loss)
        metrics["round_counter"] = float(round_counter)

    metrics["train_loss"] = float(loss)
    metrics["num-examples"] = float(n_examples)

    # Note this protects the update against a curious SERVER. It does NOT
    # affect the GAN attack, because that adversary is a CLIENT: it holds the
    # decryption key and needs only the shared model, which every participant
    # must receive in usable form. Running with and without this flag is meant
    # to demonstrate exactly that invariance.
    if bool(cfg.get("use-encryption", False)):
        old = msg.content["arrays"].to_torch_state_dict()
        new = model.state_dict()
        keys = list(old.keys())
        flat = torch.cat([(new[k] - old[k]).flatten() for k in keys]).cpu().numpy()

        pk = PaillierCrypto.deserialize_public_key(str(cfg["paillier-pubkey"]))
        crypto = PaillierCrypto(key_length=pk.n.bit_length())
        crypto.public_key = pk
        n_enc = min(len(flat), int(cfg.get("max-encrypt-values", 500)))
        ct = PaillierCrypto.serialize_encrypted(crypto.encrypt_vector(flat[:n_enc]))

        metrics["encrypted"] = 1.0
        metrics["n_encrypted"] = float(n_enc)
        content = RecordDict({
            "arrays": ArrayRecord({}),                       # no plaintext head
            "enc_delta": ConfigRecord({
                "ciphertexts": ct,
                "n_encrypted": n_enc,
                "n_total": int(len(flat)),
            }),
            "plain_tail": ArrayRecord(
                {"tail": torch.tensor(flat[n_enc:], dtype=torch.float32)}
            ),
            "metrics": MetricRecord(metrics),
        })
        return Message(content=content, reply_to=msg)

    metrics["encrypted"] = 0.0
    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord(metrics),
    })
    return Message(content=content, reply_to=msg)


@app.train()
def train(msg: Message, context: Context) -> Message:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = msg.content.get("config", ConfigRecord({}))

    if bool(cfg.get("use-gan", False)) or is_gan_state(
            msg.content["arrays"].to_torch_state_dict()):
        return gan_train(msg, context, cfg)

    # DLG / iDLG baseline: single-sample gradient, server is adversary
    use_encryption = bool(cfg.get("use-encryption", False))
    max_encrypt = int(cfg.get("max-encrypt-values", 2000))
    partition_id = int(context.node_config["partition-id"])

    net = LeNet(num_classes=100).to(device)
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    net.eval()

    dst = load_cifar100()
    img_index = partition_id
    gt_data = tp(dst[img_index][0]).to(device).view(1, 3, 32, 32)
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device).view(1)
    gt_onehot_label = label_to_onehot(gt_label, num_classes=100)

    out = net(gt_data)
    y_loss = cross_entropy_for_onehot(out, gt_onehot_label)
    dy_dx = torch.autograd.grad(y_loss, net.parameters())
    gradients = [g.detach() for g in dy_dx]
    flat_grad = torch.cat([g.flatten() for g in gradients]).cpu().numpy()

    metrics = MetricRecord({
        "partition_id": partition_id,
        "img_index": img_index,
        "label": int(gt_label.item()),
        "num-examples": 1,
        "encrypted": int(use_encryption),
    })

    if use_encryption:
        pk = PaillierCrypto.deserialize_public_key(str(cfg["paillier-pubkey"]))
        crypto = PaillierCrypto(key_length=pk.n.bit_length())
        crypto.public_key = pk
        n_enc = min(len(flat_grad), max_encrypt)
        serialized = PaillierCrypto.serialize_encrypted(
            crypto.encrypt_vector(flat_grad[:n_enc])
        )
        enc_record = ConfigRecord({
            "ciphertexts": serialized,
            "n_encrypted": n_enc,
            "n_total": len(flat_grad),
        })
        content = RecordDict({
            "arrays": ArrayRecord({}),          # no plaintext gradient!
            "encrypted_grad": enc_record,
            "metrics": metrics,
        })
        return Message(content=content, reply_to=msg)

    original_state = msg.content["arrays"].to_torch_state_dict()
    updated_state = {}
    for i, (key, param) in enumerate(original_state.items()):
        updated_state[key] = param - gradients[i]

    content = RecordDict({
        "arrays": ArrayRecord(updated_state),
        "metrics": metrics,
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = msg.content.get("config", ConfigRecord({}))

    state = msg.content["arrays"].to_torch_state_dict()
    if bool(cfg.get("use-gan", False)) or is_gan_state(state):
        if "mnist-root" in cfg:
            os.environ.setdefault("MNIST_ROOT", str(cfg["mnist-root"]))
        partition_id = int(context.node_config["partition-id"])
        is_adversary = (partition_id == int(cfg.get("gan-adversary-id", 1)))
        model = CNNClassifier().to(device)
        model.load_state_dict(state)
        _, test_loader = load_partition(is_victim=not is_adversary,
                                        batch_size=int(cfg.get("batch-size", 64)))
        loss, acc = evaluate_classifier(model, test_loader, device)
        metrics = MetricRecord({"accuracy": float(acc), "loss": float(loss),
                                "num-examples": len(test_loader.dataset)})
        return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)

    net = LeNet(num_classes=100).to(device)
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    net.eval()

    dst = load_cifar100()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for i in range(min(256, len(dst))):
            img = tp(dst[i][0]).unsqueeze(0).to(device)
            lab = torch.tensor([dst[i][1]], dtype=torch.long, device=device)
            out = net(img)
            total_loss += F.cross_entropy(out, lab).item()
            correct += (out.argmax(1) == lab).sum().item()
            total += 1

    metrics = MetricRecord({"accuracy": correct / total,
                            "loss": total_loss / total, "num-examples": total})
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
