"""Deterministic random substreams derived from one root seed.

Every random draw in the project takes a generator produced here. A substream is identified by a
root seed and a tuple of labels (ints or strings). Integer labels map onto NumPy's spawn keys, so
``substream(s, i)`` equals ``np.random.SeedSequence(s).spawn(i + 1)[i]`` for ``i >= 0``. String
labels are hashed with SHA-256 into a key range that integer labels below ``2**128`` never reach.
The mapping uses only SeedSequence and PCG64, both of which are platform independent.
"""

from __future__ import annotations

import hashlib

import numpy as np

Label = int | str

_STR_TAG = 1 << 128


def _label_word(label: Label) -> int:
    if isinstance(label, bool) or not isinstance(label, (int, np.integer, str)):
        raise TypeError(f"seed labels must be int or str, got {type(label).__name__}")
    if isinstance(label, str):
        digest = hashlib.sha256(label.encode("utf-8")).digest()
        return _STR_TAG + int.from_bytes(digest[:16], "little")
    value = int(label)
    if value < 0 or value >= _STR_TAG:
        raise ValueError(f"integer seed labels must lie in [0, 2**128), got {value}")
    return value


def seed_sequence(root: int, *labels: Label) -> np.random.SeedSequence:
    """Return the SeedSequence of the substream ``(root, *labels)``."""
    if isinstance(root, bool) or int(root) < 0:
        raise ValueError(f"root seed must be a non-negative int, got {root!r}")
    return np.random.SeedSequence(int(root), spawn_key=tuple(_label_word(lb) for lb in labels))


def substream(root: int, *labels: Label) -> np.random.Generator:
    """Return a PCG64 generator for the substream ``(root, *labels)``."""
    return np.random.Generator(np.random.PCG64(seed_sequence(root, *labels)))


def derive_seed(root: int, *labels: Label) -> int:
    """Return a 63-bit integer seed for the substream, for consumers that take a plain int
    (torch generators, C++ binaries)."""
    words = seed_sequence(root, *labels).generate_state(2, dtype=np.uint32)
    return (int(words[0]) | (int(words[1]) << 32)) & ((1 << 63) - 1)


def torch_generator(root: int, *labels: Label, device: str = "cpu"):
    """Return a ``torch.Generator`` on ``device`` seeded from the substream."""
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(derive_seed(root, *labels))
    return gen
