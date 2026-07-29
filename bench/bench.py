"""Benchmarks against the equivalent Nevergrad optimizers."""

from __future__ import annotations

import math
import os
import platform
import sys
import time
import warnings
from collections.abc import Callable

import nevergrad as ng
import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojonevergrad as mng  # noqa: E402
from mojonevergrad import _lib  # noqa: E402


def best_time(function: Callable[[], object], repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def prepared(name: str, implementation, dimension: int):
    optimizer = implementation(dimension, budget=10_000)
    optimizer.parametrization.random_state.seed(7)
    population = 40 if name == "PSO" else 30 if name in ("DE", "TwoPointsDE") else 1
    for _ in range(population):
        candidate = optimizer.ask()
        optimizer.tell(candidate, float(np.dot(candidate.value, candidate.value)))
    return optimizer


def ask_case(name: str, dimension: int, calls: int):
    upstream = prepared(name, getattr(ng.optimizers, name), dimension)
    mojo = prepared(name, getattr(mng.optimizers, name), dimension)

    def run(optimizer):
        def execute():
            for _ in range(calls):
                optimizer.ask()

        return execute

    run(mojo)()
    return best_time(run(mojo)) / calls, best_time(run(upstream)) / calls


def minimize_case(dimension: int, budget: int):
    def objective(value):
        return float(np.dot(value - 0.25, value - 0.25))

    def run(implementation):
        def execute():
            optimizer = implementation(dimension, budget=budget)
            optimizer.parametrization.random_state.seed(9)
            return optimizer.minimize(objective)

        return execute

    return best_time(run(mng.optimizers.DE), repeat=3), best_time(
        run(ng.optimizers.DE), repeat=3
    )


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    warnings.filterwarnings("ignore")
    _lib.lib()
    cases = [
        ("OnePlusOne.ask, 1,000,000d", lambda: ask_case("OnePlusOne", 1_000_000, 8)),
        ("DE.ask, 100,000d", lambda: ask_case("DE", 100_000, 8)),
        ("TwoPointsDE.ask, 100,000d", lambda: ask_case("TwoPointsDE", 100_000, 8)),
        ("PSO.ask, 100,000d", lambda: ask_case("PSO", 100_000, 8)),
        ("DE.minimize sphere, 64d/300 evals", lambda: minimize_case(64, 300)),
    ]
    print(f"Machine: {cpu_model()}; {platform.system()} {platform.release()}")
    print(f"Python {platform.python_version()}; Nevergrad {ng.__version__}")
    print()
    print("| case | mojo-nevergrad | nevergrad | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, build in cases:
        mojo_time, upstream_time = build()
        ratio = upstream_time / mojo_time
        result = f"{ratio:.2f}x faster" if ratio >= 1.0 else f"{1.0 / ratio:.2f}x slower"
        print(
            f"| {name} | {mojo_time * 1e3:.3f} ms | "
            f"{upstream_time * 1e3:.3f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
