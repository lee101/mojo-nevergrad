"""ctypes access to the compiled Mojo optimizer kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJONEVERGRAD_LIB") or os.path.join(
    ROOT, "dist", "libmojo-nevergrad.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mng_scale": ([I, I, I, F], None),
    "mng_de_binomial": ([I, I, I, I, I, I, I, F, F, F, I], None),
    "mng_de_twopoints": ([I, I, I, I, I, I, F, F, I, I, I], None),
    "mng_pso_update": ([I, I, I, I, I, I, I, I, I, F, F, F], None),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJONEVERGRAD_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [
            pixi,
            "run",
            "--manifest-path",
            os.path.join(ROOT, "pixi.toml"),
            "mojo",
        ]
    raise BuildError("mojo not found; set MOJONEVERGRAD_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    if os.environ.get("MOJONEVERGRAD_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not os.path.isdir(SRC):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(
            f"no Mojo sources at {SRC} and no shared library at {LIB}; "
            "set MOJONEVERGRAD_LIB to a built library"
        )
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(os.path.getmtime(source) for source in sources):
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        os.path.join(SRC, "kernels.mojo"),
        "-o",
        LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(value, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(value)
    loses_integer_precision = source.dtype.kind in "iu" and source.dtype.itemsize > 4
    if loses_integer_precision or not np.can_cast(
        source.dtype, np.float64, casting="safe"
    ):
        raise TypeError(f"cannot safely convert dtype {source.dtype} to float64")
    if copy:
        return np.array(source, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(source, dtype=np.float64)


def addr(array: np.ndarray) -> int:
    return array.ctypes.data


def _input(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float64:
        raise TypeError(f"{name} must have dtype float64, got {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return array


def _inputs(*values: tuple[object, str]) -> list[np.ndarray]:
    arrays = [_input(value, name) for value, name in values]
    size = arrays[0].size
    if any(array.size != size for array in arrays[1:]):
        lengths = ", ".join(
            f"{name}={array.size}" for array, (_, name) in zip(arrays, values)
        )
        raise ValueError(f"array sizes must match ({lengths})")
    return arrays


def scale(noise: np.ndarray, sigma: float) -> np.ndarray:
    noise = _input(noise, "noise")
    result = np.empty_like(noise)
    if noise.size:
        lib().mng_scale(addr(noise), addr(result), noise.size, sigma)
    return result


def de_binomial(
    parent: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    best: np.ndarray,
    random: np.ndarray,
    f1: float,
    f2: float,
    crossover: float,
    forced: int,
) -> np.ndarray:
    parent, first, second, best, random = _inputs(
        (parent, "parent"),
        (first, "first"),
        (second, "second"),
        (best, "best"),
        (random, "random"),
    )
    if not 0 <= forced < parent.size:
        raise ValueError(f"forced index {forced} is outside size {parent.size}")
    result = np.empty_like(parent)
    lib().mng_de_binomial(
        addr(parent),
        addr(first),
        addr(second),
        addr(best),
        addr(random),
        addr(result),
        parent.size,
        f1,
        f2,
        crossover,
        forced,
    )
    return result


def de_twopoints(
    parent: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    best: np.ndarray,
    f1: float,
    f2: float,
    lower: int,
    upper: int,
    parent_inside: bool,
) -> np.ndarray:
    parent, first, second, best = _inputs(
        (parent, "parent"),
        (first, "first"),
        (second, "second"),
        (best, "best"),
    )
    if not 0 <= lower <= upper <= parent.size:
        raise ValueError(
            f"invalid crossover bounds [{lower}, {upper}) for size {parent.size}"
        )
    result = np.empty_like(parent)
    if parent.size:
        lib().mng_de_twopoints(
            addr(parent),
            addr(first),
            addr(second),
            addr(best),
            addr(result),
            parent.size,
            f1,
            f2,
            lower,
            upper,
            int(parent_inside),
        )
    return result


def pso_update(
    x: np.ndarray,
    speed: np.ndarray,
    parent_best: np.ndarray,
    global_best: np.ndarray,
    rp: np.ndarray,
    rg: np.ndarray,
    omega: float,
    phip: float,
    phig: float,
) -> tuple[np.ndarray, np.ndarray]:
    x, speed, parent_best, global_best, rp, rg = _inputs(
        (x, "x"),
        (speed, "speed"),
        (parent_best, "parent_best"),
        (global_best, "global_best"),
        (rp, "rp"),
        (rg, "rg"),
    )
    new_speed = np.empty_like(x)
    boxed = np.empty_like(x)
    if x.size:
        lib().mng_pso_update(
            addr(x),
            addr(speed),
            addr(parent_best),
            addr(global_best),
            addr(rp),
            addr(rg),
            addr(new_speed),
            addr(boxed),
            x.size,
            omega,
            phip,
            phig,
        )
    return new_speed, boxed


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
