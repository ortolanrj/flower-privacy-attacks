"""Additively Homomorphic Encryption (Paillier) for privacy-preserving FL.

Implements the gradients-encrypted aggregation defence of
Phong et al., "Privacy-Preserving Deep Learning via Additively
Homomorphic Encryption" (2017), https://eprint.iacr.org/2017/715.pdf

Core idea (Section 4 of the paper):
    Each participant encrypts its gradient update E(-alpha * G_local) before
    sending it to the server. The honest-but-curious server aggregates the
    ciphertexts homomorphically:

        E(W) := E(W) + E(-alpha * G_local)

    By the additive homomorphic property of Paillier:

        E(a) + E(b) = E(a + b)     [Property 1, Eq. (8) in the paper]

    so the decrypted result equals the correct plaintext aggregate, while the
    server never observes any plaintext gradient. Since the DLG attack
    requires the plaintext gradient G to solve

        argmin_{x',y'} || dW(x',y') - G ||^2,

    and the server only holds E(G), the attack cannot be instantiated.
"""

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from phe import paillier


@dataclass
class EncryptionStats:
    """Bookkeeping for the efficiency analysis."""
    key_length: int
    num_values: int
    keygen_time: float = 0.0
    encrypt_time: float = 0.0
    aggregate_time: float = 0.0
    decrypt_time: float = 0.0
    plaintext_bytes: int = 0
    ciphertext_bytes: int = 0

    @property
    def overhead_factor(self) -> float:
        if self.plaintext_bytes == 0:
            return float("nan")
        return self.ciphertext_bytes / self.plaintext_bytes


class PaillierCrypto:
    """Wrapper around the Paillier cryptosystem for gradient vectors.

    The participants jointly hold the secret key; the server holds only the
    public key. In this simulation a single PaillierCrypto instance models the
    shared key material of the participant group.
    """

    def __init__(self, key_length: int = 1024, precision: float = 1e-6):
        """
        Args:
            key_length: bit length of the Paillier modulus n
            precision: fixed-point precision for encoding real-valued gradients.
        """
        self.key_length = key_length
        self.precision = precision
        self.public_key: Optional[paillier.PaillierPublicKey] = None
        self.secret_key: Optional[paillier.PaillierPrivateKey] = None
        self.stats = EncryptionStats(key_length=key_length, num_values=0)

    # Key generation
    def generate_keys(self) -> None:
        """Generate a Paillier key pair (pk, sk).

        n = p*q with p, q large primes; pk = (n, g), sk = (lambda, mu).
        """
        t0 = time.perf_counter()
        self.public_key, self.secret_key = paillier.generate_paillier_keypair(
            n_length=self.key_length
        )
        self.stats.keygen_time = time.perf_counter() - t0

    # Encryption  E(m) = g^m * r^n mod n^2
    def encrypt_vector(self, values: np.ndarray) -> List[paillier.EncryptedNumber]:
        """Encrypt a flat array of real numbers element-wise.

        Each gradient component g_i is encoded as a fixed-point number and
        encrypted independently, producing E(g_i).
        """
        if self.public_key is None:
            raise RuntimeError("Keys not generated. Call generate_keys() first.")

        flat = np.asarray(values, dtype=np.float64).ravel()
        t0 = time.perf_counter()
        encrypted = [
            self.public_key.encrypt(float(v), precision=self.precision)
            for v in flat
        ]
        self.stats.encrypt_time += time.perf_counter() - t0
        self.stats.num_values = len(flat)
        self.stats.plaintext_bytes = flat.nbytes
        # Each ciphertext is an element of Z_{n^2}, i.e. ~2*key_length bits
        self.stats.ciphertext_bytes = len(flat) * (2 * self.key_length // 8)
        return encrypted

    # Homomorphic aggregation:  E(a) + E(b) = E(a + b)
    def aggregate(
        self,
        encrypted_vectors: List[List[paillier.EncryptedNumber]],
    ) -> List[paillier.EncryptedNumber]:
        if not encrypted_vectors:
            return []
        n_clients = len(encrypted_vectors)
        n_params = len(encrypted_vectors[0])

        t0 = time.perf_counter()
        aggregated = []
        for j in range(n_params):
            acc = encrypted_vectors[0][j]
            for i in range(1, n_clients):
                acc = acc + encrypted_vectors[i][j]  # ciphertext addition
            aggregated.append(acc)
        self.stats.aggregate_time += time.perf_counter() - t0
        return aggregated

    # Decryption  Dec(c) = L(c^lambda mod n^2) * mu mod n
    def decrypt_vector(
        self,
        encrypted: List[paillier.EncryptedNumber],
        shape: Optional[tuple] = None,
    ) -> np.ndarray:
        """Decrypt an encrypted vector back to a NumPy array."""
        if self.secret_key is None:
            raise RuntimeError("Secret key unavailable. Cannot decrypt.")

        t0 = time.perf_counter()
        decrypted = np.array(
            [self.secret_key.decrypt(c) for c in encrypted], dtype=np.float64
        )
        self.stats.decrypt_time += time.perf_counter() - t0
        if shape is not None:
            decrypted = decrypted.reshape(shape)
        return decrypted

    # Adversary's view without the secret key (for the demonstration)
    def ciphertext_as_raw_numbers(
        self,
        encrypted: List[paillier.EncryptedNumber],
    ) -> np.ndarray:
        """Return the raw ciphertext integers as floats.

        This models what an honest-but-curious server *physically holds* when
        it cannot decrypt: a vector of huge integers in Z_{n^2} that, under the
        DCRA / IND-CPA assumption, is computationally indistinguishable from
        random. Feeding these into DLG yields noise — demonstrating the
        defence. (For numerical stability we map them to a bounded range.)
        """
        raw = np.array(
            [float(c.ciphertext(be_secure=False) % (2**53)) for c in encrypted],
            dtype=np.float64,
        )
        # Normalise to a gradient-like scale so the DLG optimiser does not
        # immediately overflow; the point is that this carries no signal.
        raw = (raw / (2**53)) - 0.5
        return raw

    # Serialization for Flower Message transport. Serialize a public key to a string (its modulus n).
    @staticmethod
    def serialize_public_key(pk: paillier.PaillierPublicKey) -> str:
        return str(pk.n)

    # Reconstruct a public key from its modulus n
    @staticmethod
    def deserialize_public_key(n_str: str) -> paillier.PaillierPublicKey:
        return paillier.PaillierPublicKey(n=int(n_str))

    # Serialise a list of ciphertexts to strings: 'ciphertext,exponent'
    @staticmethod
    def serialize_encrypted(encrypted: List[paillier.EncryptedNumber]) -> List[str]:
        return [
            f"{c.ciphertext(be_secure=False)},{c.exponent}" for c in encrypted
        ]

    # Reconstruct ciphertexts from their string representation.
    @staticmethod
    def deserialize_encrypted(
        serialized: List[str],
        pk: paillier.PaillierPublicKey,
    ) -> List[paillier.EncryptedNumber]:
        result = []
        for s in serialized:
            ct_str, exp_str = s.split(",")
            result.append(
                paillier.EncryptedNumber(pk, int(ct_str), int(exp_str))
            )
        return result
