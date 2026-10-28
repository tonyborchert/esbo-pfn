import numpy as np
import pytest
import torch

from esbopfn.util.seeds import derive_seed, substream, torch_generator


def test_same_labels_same_stream():
    a = substream(7, "train", 3).random(5)
    b = substream(7, "train", 3).random(5)
    np.testing.assert_array_equal(a, b)


def test_label_sensitivity():
    base = substream(7, "train", 3).random(4)
    for other in [substream(8, "train", 3), substream(7, "val", 3), substream(7, "train", 4),
                  substream(7, 3, "train"), substream(7, "train"), substream(7, "train", "3")]:
        assert not np.array_equal(base, other.random(4))


def test_int_labels_match_numpy_spawn():
    spawned = np.random.SeedSequence(11).spawn(3)[2]
    ref = np.random.Generator(np.random.PCG64(spawned)).random(3)
    np.testing.assert_array_equal(substream(11, 2).random(3), ref)


def test_known_values_are_platform_independent():
    # PCG64 and SeedSequence are specified bit for bit, so these values hold everywhere.
    assert substream(0).integers(0, 2**31, size=3).tolist() == \
        np.random.default_rng(0).integers(0, 2**31, size=3).tolist()
    assert substream(0, "a", 1).integers(0, 2**31, size=3).tolist() == \
        [301604881, 2046604230, 1418264178]
    assert derive_seed(0, "a", 1) == 7888081277333142191
    assert 0 <= derive_seed(123, "x", 5) < 2**63


def test_torch_generator_deterministic():
    a = torch.rand(4, generator=torch_generator(1, "init"))
    b = torch.rand(4, generator=torch_generator(1, "init"))
    c = torch.rand(4, generator=torch_generator(1, "other"))
    assert torch.equal(a, b) and not torch.equal(a, c)


@pytest.mark.parametrize("bad", [-1, 1.5, True, None])
def test_bad_labels(bad):
    with pytest.raises((TypeError, ValueError)):
        substream(0, bad)
