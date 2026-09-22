"""ServerApp — malicious server with Paillier-AHE defence.

Uses the Flower Message API (@app.main()).

The server generates a Paillier keypair and distributes the PUBLIC key to the
clients in the train config. (In a real deployment the participants jointly
own (pk, sk) and the server holds only pk; here the simulation generates them
centrally for convenience, but the server's strategy never uses sk to attack.)

Behaviour of the malicious (honest-but-curious) server:
  * use-encryption = False:
        clients send plaintext gradients -> the server runs DLG and
        reconstructs a victim's private image (attack succeeds).
  * use-encryption = True:
        clients send only E(G) -> the server's view of the gradient is the
        ciphertext integers (IND-CPA: indistinguishable from random). Running
        DLG on this view yields noise (attack fails -> defended). The server
        can still aggregate homomorphically (accuracy preserved, Theorem 2).
"""

import random
from typing import Iterable, Optional

import numpy as np
import torch

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from flower_attacks.crypto import PaillierCrypto
from flower_attacks.gan_attack import CNNClassifier, mnist_root, prepare_mnist
from flower_attacks.task import (
    LeNet,
    dlg_attack_from_gradients,
    load_cifar100,
    tp,
    tt,
    weights_init,
)


class MaliciousFedAvg(FedAvg):
    """FedAvg whose server attempts DLG on a random victim each round."""

    def __init__(self, dlg_iterations=300, use_encryption=False,
                 crypto: Optional[PaillierCrypto] = None,
                 use_idlg=False, benchmark=False, **kwargs):
        super().__init__(**kwargs)
        self.dlg_iterations = dlg_iterations
        self.use_encryption = use_encryption
        self.crypto = crypto
        self.use_idlg = use_idlg
        self.benchmark = benchmark
        self._server_arrays: Optional[ArrayRecord] = None
        self.bench_rows: list[dict] = []

    def configure_train(self, server_round, arrays, config, grid):
        self._server_arrays = arrays
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        replies_list = list(replies)
        if not replies_list:
            return super().aggregate_train(server_round, iter(replies_list))

        # ---- Randomly pick a victim ----
        victim = replies_list[random.randint(0, len(replies_list) - 1)]
        m = victim.content.get("metrics")
        victim_id = int(m["partition_id"]) if m else "?"
        victim_label = int(m["label"]) if m else "?"

        print(f"\n{'#'*60}")
        print(f"  SERVER round {server_round}: attacking client {victim_id} "
              f"(label {victim_label})")
        print(f"  Attack method: {'iDLG' if self.use_idlg else 'DLG'}")
        print(f"  Encryption enabled: {self.use_encryption}")
        print(f"{'#'*60}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = LeNet().to(device)
        server_state = self._server_arrays.to_torch_state_dict()
        net.load_state_dict({k: v.to(device) for k, v in server_state.items()})
        net.eval()

        # Ground truth for visual comparison (simulation only)
        try:
            dst = load_cifar100()
            img_idx = int(m["img_index"]) if m else 0
            gt_image = tt(tp(dst[img_idx][0]))
        except Exception:
            gt_image = None

        if self.use_encryption:
            # ---- Server holds only E(G): reconstruct its ciphertext view ----
            enc_rec = victim.content.get("encrypted_grad")
            serialized = list(enc_rec["ciphertexts"])
            n_total = int(enc_rec["n_total"])
            pk = self.crypto.public_key
            encrypted = PaillierCrypto.deserialize_encrypted(serialized, pk)

            # The server CANNOT decrypt. Its usable view is the ciphertext
            # integers, which are indistinguishable from random (IND-CPA).
            raw_subset = self.crypto.ciphertext_as_raw_numbers(encrypted)
            rng = np.random.RandomState(server_round)
            full_view = rng.uniform(raw_subset.min(), raw_subset.max(),
                                    size=n_total).astype(np.float64)
            full_view[:len(raw_subset)] = raw_subset

            # Shape back into gradient tensors
            server_gradients = []
            offset = 0
            for p in net.parameters():
                size = p.numel()
                chunk = full_view[offset:offset + size]
                server_gradients.append(
                    torch.tensor(chunk, dtype=torch.float32, device=device).reshape(p.shape)
                )
                offset += size
            print("  Server view = ciphertexts (no secret key). DLG will fail.")
        else:
            # ---- Server recovers plaintext gradient from the update ----
            victim_state = victim.content["arrays"].to_torch_state_dict()
            server_gradients = [
                (server_state[k] - victim_state[k]).to(device) for k in server_state
            ]
            print("  Server view = plaintext gradient. DLG will succeed.")

        # ---- Benchmark mode: run BOTH variants on the SAME gradients ----
        if self.benchmark:
            row = {"round": server_round, "victim": victim_id,
                   "true_label": victim_label}
            for variant in (False, True):
                stats = {}
                dlg_attack_from_gradients(
                    net=net,
                    original_dy_dx=server_gradients,
                    input_shape=(1, 3, 32, 32),
                    num_classes=100,
                    num_iterations=self.dlg_iterations,
                    use_idlg=variant,
                    save_dir="dlg_results",
                    client_id=f"bench_{'idlg' if variant else 'dlg'}_r{server_round}",
                    gt_image=gt_image,
                    stats_out=stats,
                    save_outputs=False,          # timing runs write no images
                )
                row["idlg" if variant else "dlg"] = stats
            self.bench_rows.append(row)
            if self.use_encryption:
                return self._server_arrays, MetricRecord({})
            return super().aggregate_train(server_round, iter(replies_list))

        # ---- Run the DLG attack ----
        method = "idlg" if self.use_idlg else "dlg"
        tag = (f"{method}_{'encrypted' if self.use_encryption else 'plaintext'}"
               f"_client_{victim_id}")
        dlg_attack_from_gradients(
            net=net,
            original_dy_dx=server_gradients,
            input_shape=(1, 3, 32, 32),
            num_classes=100,
            num_iterations=self.dlg_iterations,
            use_idlg=self.use_idlg,
            save_dir="dlg_results",
            client_id=tag,
            gt_image=gt_image,
        )

        # Normal aggregation continues (skip for encrypted, since arrays empty)
        if self.use_encryption:
            return self._server_arrays, MetricRecord({})
        return super().aggregate_train(server_round, iter(replies_list))


    def print_benchmark(self):
        """Summarise the in-federation DLG vs iDLG comparison."""
        if not self.bench_rows:
            return
        rows = self.bench_rows
        hdr = (f"{'round':>5} {'victim':>7} {'true':>5} | "
               f"{'DLG lbl':>8} {'conv':>5} {'s':>7} {'ms/it':>7} | "
               f"{'iDLG lbl':>9} {'conv':>5} {'s':>7} {'ms/it':>7}")
        print("\n" + "=" * len(hdr))
        print("  DLG vs iDLG — measured inside the Flower federation")
        print("  (both attacks run on the SAME intercepted gradients)")
        print("=" * len(hdr))
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            d, i, tl = r["dlg"], r["idlg"], r["true_label"]
            dm = "" if d["recovered_label"] == tl else "*"
            im = "" if i["recovered_label"] == tl else "*"
            print(f"{r['round']:>5} {r['victim']:>7} {tl:>5} | "
                  f"{d['recovered_label']:>7}{dm:<1} "
                  f"{str(d['converged_at'] or '-'):>5} {d['elapsed_s']:>7.2f} "
                  f"{d['ms_per_iter']:>7.1f} | "
                  f"{i['recovered_label']:>8}{im:<1} "
                  f"{str(i['converged_at'] or '-'):>5} {i['elapsed_s']:>7.2f} "
                  f"{i['ms_per_iter']:>7.1f}")
        print("-" * len(hdr))

        def mean(key, variant):
            vals = [r[variant][key] for r in rows if r[variant][key] is not None]
            return sum(vals) / len(vals) if vals else float("nan")

        d_ok = sum(1 for r in rows if r["dlg"]["recovered_label"] == r["true_label"])
        i_ok = sum(1 for r in rows if r["idlg"]["recovered_label"] == r["true_label"])
        dt, it = mean("elapsed_s", "dlg"), mean("elapsed_s", "idlg")
        print(f"  label accuracy   : DLG {d_ok}/{len(rows)}    iDLG {i_ok}/{len(rows)}")
        print(f"  mean wall-clock  : DLG {dt:.2f}s   iDLG {it:.2f}s   "
              f"({abs(dt - it) / max(dt, it) * 100:.1f}% apart)")
        print(f"  mean ms/iteration: DLG {mean('ms_per_iter','dlg'):.1f}   "
              f"iDLG {mean('ms_per_iter','idlg'):.1f}")
        dc, ic = mean("converged_at", "dlg"), mean("converged_at", "idlg")
        if dc == dc and ic == ic:            # both non-NaN
            print(f"  mean iters to converge: DLG {dc:.1f}   iDLG {ic:.1f}")
        else:
            print("  mean iters to converge: n/a (no run reached the threshold; "
                  "raise dlg-iterations)")
        print("  (* = wrong label recovered)")
        print("=" * len(hdr))


class HonestFedAvg(FedAvg):
    """Plain FedAvg for the GAN attack scenario, with optional PHE aggregation.

    In Hitaj et al. the adversary is a *participant*, and the paper states it
    "does not compromise the central parameter server". The server therefore
    performs no attack here.

    When `use_encryption` is set, clients send their model update encrypted
    under Paillier and this strategy aggregates the ciphertexts homomorphically
    -- the server never sees an individual plaintext update. The purpose is to
    show that this makes NO difference to the GAN attack, since that adversary
    is a client that holds the key and only needs the shared model.
    """

    def __init__(self, use_encryption=False, crypto=None, **kwargs):
        super().__init__(**kwargs)
        self.use_encryption = use_encryption
        self.crypto = crypto
        self._server_arrays = None

    def configure_train(self, server_round, arrays, config, grid):
        self._server_arrays = arrays
        config["server-round"] = server_round
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        if not self.use_encryption or not replies or self._server_arrays is None:
            return super().aggregate_train(server_round, iter(replies))

        # ---- homomorphic aggregation of the encrypted deltas ----
        old_state = self._server_arrays.to_torch_state_dict()
        keys = list(old_state.keys())
        pk = self.crypto.public_key

        enc_vectors, tails, n_clients = [], [], 0
        n_enc = n_total = 0
        for r in replies:
            rec = r.content.get("enc_delta")
            if rec is None:
                continue
            n_enc = int(rec["n_encrypted"])
            n_total = int(rec["n_total"])
            enc_vectors.append(
                PaillierCrypto.deserialize_encrypted(list(rec["ciphertexts"]), pk)
            )
            tails.append(
                r.content["plain_tail"].to_torch_state_dict()["tail"].numpy()
            )
            n_clients += 1

        if n_clients == 0:
            return super().aggregate_train(server_round, iter(replies))

        # E(d_1) + ... + E(d_n) = E(sum d_i)  -- server uses only the PUBLIC key
        agg_enc = self.crypto.aggregate(enc_vectors)
        # In a real deployment the participants (key holders) decrypt. In this
        # simulation the same object holds sk; the server still never observes
        # any individual plaintext update, only the aggregate.
        agg_head = self.crypto.decrypt_vector(agg_enc) / n_clients
        agg_tail = np.mean(np.stack(tails, axis=0), axis=0) if n_total > n_enc \
            else np.zeros(0, dtype=np.float64)
        avg_delta = np.concatenate([agg_head, agg_tail])

        print(f"  [server] round {server_round}: homomorphic aggregation over "
              f"{n_clients} clients ({n_enc}/{n_total} values encrypted); "
              f"no individual plaintext update observed")

        # rebuild the global model: W <- W + mean(delta)
        new_state, offset = {}, 0
        for k in keys:
            numel = old_state[k].numel()
            chunk = torch.tensor(avg_delta[offset:offset + numel],
                                 dtype=old_state[k].dtype).reshape(old_state[k].shape)
            new_state[k] = old_state[k] + chunk
            offset += numel

        return ArrayRecord(new_state), MetricRecord({"encrypted_aggregation": 1.0})


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = int(context.run_config.get("num-server-rounds", 1))
    dlg_iterations = int(context.run_config.get("dlg-iterations", 300))
    use_encryption = bool(context.run_config.get("use-encryption", False))
    use_idlg = bool(context.run_config.get("use-idlg", False))
    benchmark = bool(context.run_config.get("benchmark-attacks", False))
    use_gan = bool(context.run_config.get("use-gan", False))
    key_length = int(context.run_config.get("paillier-key-length", 1024))
    max_encrypt = int(context.run_config.get("max-encrypt-values", 2000))

    # Initialise model (notebook init)
    torch.manual_seed(50)
    # The GAN attack adds one output neuron for the adversary's 'fake' class
    num_classes = 101 if use_gan else 100
    global_model = CNNClassifier() if use_gan else LeNet(num_classes=num_classes)
    if not use_gan:
        # uniform(-0.5, 0.5) reproduces the DLG/iDLG reference setup, but it is
        # a poor initialisation for a model that must actually learn. The GAN
        # attack depends on the shared model becoming ACCURATE, so in that mode
        # we keep PyTorch's default (Kaiming-uniform) initialisation.
        global_model.apply(weights_init)
    arrays = ArrayRecord(global_model.state_dict())

    # Build the train config; if encryption is on, generate and share keys
    train_cfg = {
        "use-encryption": use_encryption,
        "max-encrypt-values": max_encrypt,
        "use-gan": use_gan,
    }
    if use_gan:
        train_cfg.update({
            "target-class": int(context.run_config.get("target-class", 4)),
            "gan-adversary-id": int(context.run_config.get("gan-adversary-id", 1)),
            "gan-latent-dim": int(context.run_config.get("gan-latent-dim", 100)),
            "gan-steps-per-round": int(context.run_config.get("gan-steps-per-round", 100)),
            "gan-batch-size": int(context.run_config.get("gan-batch-size", 64)),
            "gan-lr": float(context.run_config.get("gan-lr", 0.02)),
            "num-injected-samples": int(context.run_config.get("num-injected-samples", 64)),
            "batch-size": int(context.run_config.get("batch-size", 64)),
            "local-epochs": int(context.run_config.get("local-epochs", 1)),
            "learning-rate": float(context.run_config.get("learning-rate", 0.001)),
            "lr-decay": float(context.run_config.get("lr-decay", 1e-7)),
            "save-every": int(context.run_config.get("save-every", 5)),
            "output-dir": str(context.run_config.get("output-dir", "outputs")),
            "num-server-rounds": num_rounds,
        })
    crypto = None
    if use_encryption:
        print(f"Generating Paillier keypair ({key_length}-bit)...")
        crypto = PaillierCrypto(key_length=key_length)
        crypto.generate_keys()
        # Share ONLY the public key with the clients
        train_cfg["paillier-pubkey"] = PaillierCrypto.serialize_public_key(
            crypto.public_key
        )
        print("Public key distributed to clients. Server cannot decrypt.")

    if use_gan:
        # Fetch MNIST ONCE here. Letting several Ray client actors download
        # concurrently into the same directory is a known cause of the
        # "File not found or corrupted" error.
        data_root = prepare_mnist()
        train_cfg["mnist-root"] = data_root

        # The adversary is a client; the server plays fair.
        print(f"GAN attack enabled (Hitaj et al., CCS 2017).")
        print(f"  Adversary  : client {train_cfg['gan-adversary-id']} (owns digits 5-9)")
        print(f"  Victim     : client 0 (owns digits 0-4)")
        print(f"  Target class: {train_cfg['target-class']} "
              f"(owned by the victim, NOT by the adversary)")
        print(f"  Server behaviour: HONEST (no gradient inversion)")
        if use_encryption:
            print("  PHE enabled: client updates encrypted under Paillier.")
            print("  NOTE: this protects the update channel against a curious")
            print("        server; it does NOT affect the GAN attack, whose")
            print("        adversary is a client holding the decryption key.")
        strategy = HonestFedAvg(use_encryption=use_encryption, crypto=crypto,
                                fraction_evaluate=1.0)
    else:
        strategy = MaliciousFedAvg(
            dlg_iterations=dlg_iterations,
            use_encryption=use_encryption,
            use_idlg=use_idlg,
            benchmark=benchmark,
            crypto=crypto,
            fraction_evaluate=0.0,
        )

    # The evaluate config is SEPARATE from the train config. Without this the
    # clients' evaluate() would not see use-gan and would build the wrong model
    # (LeNet instead of CNNClassifier), producing a state_dict key mismatch.
    eval_cfg = {
        "use-gan": use_gan,
        "gan-adversary-id": int(train_cfg.get("gan-adversary-id", 1)),
        "batch-size": int(train_cfg.get("batch-size", 64)),
    }
    if use_gan and "mnist-root" in train_cfg:
        eval_cfg["mnist-root"] = train_cfg["mnist-root"]

    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(train_cfg),
        evaluate_config=ConfigRecord(eval_cfg),
        num_rounds=num_rounds,
    )

    if not use_gan and getattr(strategy, "benchmark", False):
        strategy.print_benchmark()

    print("\nDone.")
