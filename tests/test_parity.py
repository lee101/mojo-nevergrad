from __future__ import annotations

import inspect
import warnings

import nevergrad as ng
import numpy as np
import pytest
from nevergrad.parametrization.parameter import Parameter

import mojonevergrad as mng
from mojonevergrad import _lib


def sphere(value: np.ndarray) -> float:
    return float(np.sum((value - 0.25) ** 2))


def paired(name: str, dimension: int, budget: int, seed: int = 12):
    upstream = getattr(ng.optimizers, name)(dimension, budget=budget)
    mojo = getattr(mng.optimizers, name)(dimension, budget=budget)
    upstream.parametrization.random_state.seed(seed)
    mojo.parametrization.random_state.seed(seed)
    return upstream, mojo


def run_paired(name: str, dimension: int, budget: int, tolerance: float):
    upstream, mojo = paired(name, dimension, budget)
    for _ in range(budget):
        expected = upstream.ask()
        actual = mojo.ask()
        np.testing.assert_allclose(
            actual.value, expected.value, rtol=tolerance, atol=tolerance
        )
        loss = sphere(expected.value)
        upstream.tell(expected, loss)
        mojo.tell(actual, loss)
    np.testing.assert_allclose(
        mojo.provide_recommendation().value,
        upstream.provide_recommendation().value,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize(
    ("name", "budget", "tolerance"),
    [
        ("OnePlusOne", 80, 0.0),
        ("DE", 100, 2e-14),
        ("TwoPointsDE", 100, 2e-14),
        ("PSO", 120, 3e-11),
    ],
)
def test_seeded_candidate_stream_matches_upstream(name, budget, tolerance):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_paired(name, dimension=8, budget=budget, tolerance=tolerance)


@pytest.mark.parametrize("name", ["OnePlusOne", "DE", "TwoPointsDE", "PSO"])
def test_public_constructor_signature_matches_upstream(name):
    ours = inspect.signature(getattr(mng.optimizers, name))
    theirs = inspect.signature(getattr(ng.optimizers, name))
    assert list(ours.parameters) == list(theirs.parameters)
    for parameter in ours.parameters:
        assert ours.parameters[parameter].default == theirs.parameters[parameter].default


@pytest.mark.parametrize("name", ["OnePlusOne", "DE", "TwoPointsDE", "PSO"])
def test_minimize_matches_upstream(name):
    upstream, mojo = paired(name, dimension=6, budget=75, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = upstream.minimize(sphere)
        actual = mojo.minimize(sphere)
    np.testing.assert_allclose(actual.value, expected.value, rtol=3e-11, atol=3e-11)
    assert mojo.num_ask == upstream.num_ask == 75
    assert mojo.num_tell == upstream.num_tell == 75


def test_candidate_is_nevergrad_parameter():
    optimizer = mng.optimizers.DE(parametrization=3, budget=10)
    candidate = optimizer.ask()
    assert isinstance(candidate, Parameter)
    assert len(candidate.args) == 1
    np.testing.assert_array_equal(candidate.args[0], candidate.value)
    assert candidate.kwargs == {}


def test_array_parametrization_and_bounds_match_upstream():
    first = ng.p.Array(shape=(5,), lower=-1.0, upper=2.0)
    second = ng.p.Array(shape=(5,), lower=-1.0, upper=2.0)
    upstream = ng.optimizers.DE(first, budget=70)
    mojo = mng.optimizers.DE(second, budget=70)
    upstream.parametrization.random_state.seed(19)
    mojo.parametrization.random_state.seed(19)
    for _ in range(70):
        expected, actual = upstream.ask(), mojo.ask()
        np.testing.assert_allclose(actual.value, expected.value, rtol=2e-14, atol=2e-14)
        assert np.all(actual.value >= -1.0) and np.all(actual.value <= 2.0)
        loss = sphere(expected.value)
        upstream.tell(expected, loss)
        mojo.tell(actual, loss)


def test_callbacks_follow_nevergrad_contract():
    events: list[tuple[str, int]] = []
    optimizer = mng.optimizers.OnePlusOne(3, budget=5)
    optimizer.register_callback("ask", lambda opt: events.append(("ask", opt.num_ask)))
    optimizer.register_callback(
        "tell", lambda opt, candidate, loss: events.append(("tell", opt.num_tell))
    )
    optimizer.minimize(sphere)
    assert [event for event, _ in events].count("ask") == 5
    assert [event for event, _ in events].count("tell") == 5


def test_de_asynchronous_tell_order_matches_upstream():
    upstream, mojo = paired("DE", dimension=7, budget=90, seed=8)
    for _ in range(6):
        expected_batch = [upstream.ask() for _ in range(5)]
        actual_batch = [mojo.ask() for _ in range(5)]
        for expected, actual in zip(expected_batch, actual_batch):
            np.testing.assert_allclose(actual.value, expected.value, rtol=2e-14, atol=2e-14)
        for expected, actual in zip(reversed(expected_batch), reversed(actual_batch)):
            loss = sphere(expected.value)
            upstream.tell(expected, loss)
            mojo.tell(actual, loss)


def test_suggest_is_honored_by_base_optimizer():
    optimizer = mng.optimizers.PSO(3, budget=4)
    optimizer.suggest([0.1, 0.2, 0.3])
    candidate = optimizer.ask()
    np.testing.assert_allclose(candidate.value, [0.1, 0.2, 0.3])
    optimizer.tell(candidate, sphere(candidate.value))


def test_registry_contains_only_covered_names():
    assert set(mng.optimizers.registry) == {
        "OnePlusOne",
        "DE",
        "TwoPointsDE",
        "PSO",
    }
    for name, constructor in mng.optimizers.registry.items():
        assert constructor is getattr(mng.optimizers, name)


@pytest.mark.parametrize("n", [10_003, 262_147])
def test_mojo_scale_kernel_matches_numpy(n):
    rng = np.random.default_rng(2)
    noise = rng.normal(size=n)
    actual = _lib.scale(noise, 0.37)
    np.testing.assert_array_equal(actual, noise * 0.37)


@pytest.mark.parametrize("n", [17, 1_000_003])
def test_mojo_scale_inplace_tail_and_large_buffer(n):
    rng = np.random.default_rng(12)
    values = rng.normal(size=n)
    expected = values * 0.37
    original_address = values.ctypes.data
    actual = _lib.scale_inplace(values, 0.37)
    assert actual.ctypes.data == original_address
    np.testing.assert_array_equal(actual, expected)


def test_mojo_de_binomial_kernel_matches_formula():
    rng = np.random.default_rng(3)
    n = 10_003
    parent, first, second, best = [rng.normal(size=n) for _ in range(4)]
    random = rng.random(n)
    forced = 177
    actual = _lib.de_binomial(
        parent,
        first,
        second,
        best,
        random,
        0.8,
        0.8,
        0.5,
        forced,
    )
    donor = parent + 0.8 * (first - second) + 0.8 * (best - parent)
    expected = np.where(random > 0.5, parent, donor)
    expected[forced] = donor[forced]
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("n", [7, 17, 10_003])
@pytest.mark.parametrize("parent_inside", [False, True])
def test_mojo_de_twopoints_simd_segments_and_tail(n, parent_inside):
    rng = np.random.default_rng(13)
    parent, first, second, best = [rng.normal(size=n) for _ in range(4)]
    lower, upper = 1, n - 2
    actual = _lib.de_twopoints(
        parent,
        first,
        second,
        best,
        0.8,
        0.8,
        lower,
        upper,
        parent_inside,
    )
    donor = parent + 0.8 * (first - second) + 0.8 * (best - parent)
    keep = np.zeros(n, dtype=bool)
    keep[lower:upper] = parent_inside
    keep[:lower] = not parent_inside
    keep[upper:] = not parent_inside
    expected = np.where(keep, parent, donor)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


def test_mojo_pso_kernel_matches_numpy_formula():
    rng = np.random.default_rng(4)
    n = 10_001
    x, speed, parent_best, global_best, rp, rg = [
        rng.random(n) for _ in range(6)
    ]
    omega = 0.5 / np.log(2.0)
    phi = 0.5 + np.log(2.0)
    actual_speed, actual_position = _lib.pso_update(
        x, speed, parent_best, global_best, rp, rg, omega, phi, phi
    )
    expected_speed = (
        omega * speed
        + phi * rp * (parent_best - x)
        + phi * rg * (global_best - x)
    )
    np.testing.assert_allclose(actual_speed, expected_speed, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(
        actual_position, np.clip(x + expected_speed, 0.0, 1.0), rtol=2e-15, atol=2e-15
    )


@pytest.mark.parametrize("n", [1, 3, 4, 7, 8, 9, 15, 16, 17])
def test_simd_and_scalar_tail_lengths(n):
    values = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    np.testing.assert_array_equal(_lib.scale(values, 0.25), values * 0.25)
    speed, boxed = _lib.pso_update(
        values,
        np.ones(n),
        values + 0.1,
        values - 0.2,
        np.full(n, 0.3),
        np.full(n, 0.4),
        0.5,
        0.6,
        0.7,
    )
    expected = 0.5 + 0.6 * 0.3 * 0.1 + 0.7 * 0.4 * -0.2
    np.testing.assert_allclose(speed, expected)
    np.testing.assert_allclose(boxed, np.clip(values + expected, 0.0, 1.0))


def test_ffi_wrappers_reject_unsafe_arrays_and_lengths():
    good = np.arange(6, dtype=np.float64)
    with pytest.raises(TypeError, match="float64"):
        _lib.scale(good.astype(np.float32), 1.0)
    with pytest.raises(ValueError, match="C-contiguous"):
        _lib.scale(good[::2], 1.0)
    readonly = good.copy()
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        _lib.scale_inplace(readonly, 1.0)
    with pytest.raises(ValueError, match="sizes must match"):
        _lib.de_binomial(good, good, good[:-1], good, good, 0.8, 0.8, 0.5, 0)
    with pytest.raises(ValueError, match="outside size"):
        _lib.de_binomial(good, good, good, good, good, 0.8, 0.8, 0.5, 6)
    with pytest.raises(TypeError, match="safely convert"):
        _lib.f64(np.array([2**63 - 1], dtype=np.int64))


def test_empty_safe_wrappers_do_not_cross_null_pointers():
    empty = np.empty(0, dtype=np.float64)
    np.testing.assert_array_equal(_lib.scale(empty, 2.0), empty)
    speed, boxed = _lib.pso_update(
        empty, empty, empty, empty, empty, empty, 0.5, 0.6, 0.7
    )
    assert speed.size == boxed.size == 0
