# mojo-nevergrad

`mojo-nevergrad` accelerates the high-dimensional update steps of selected
[Nevergrad](https://facebookresearch.github.io/nevergrad/) derivative-free
optimizers with compiled Mojo kernels.

For the covered subset, change only the import:

```python
import mojonevergrad as ng

optimizer = ng.optimizers.DE(parametrization=8, budget=300)
recommendation = optimizer.minimize(
    lambda x: float(((x - 0.25) ** 2).sum())
)
print(recommendation.value)
```

Candidates are real Nevergrad `Parameter` objects. Tests cover the public
constructors, `ask`/`tell`, `provide_recommendation`, `minimize`, `suggest`,
callbacks, bounded arrays, and asynchronous DE tell order.

## Coverage

The continuous default variants of these optimizer names are covered:

- `OnePlusOne`: Gaussian mutation and one-fifth step-size adaptation
- `DE`: current-to-best differential mutation with binomial crossover
- `TwoPointsDE`: current-to-best differential mutation with two-point crossover
- `PSO`: the default arctangent-bounded particle swarm

Seeded tests compare every proposed point and the final recommendation with the
pinned Nevergrad release. The Mojo and upstream implementations consume random
values in the same order.

This is not the full Nevergrad optimizer registry. `NGOpt`, CMA variants, Bayesian
optimization, discrete mutations, one-shot samplers, DE's configured crossover and
initialization variants, and algorithm-specific multiobjective DE adaptation are not
ported. The port does not claim parity for multiobjective optimization, custom
configured optimizer variants, discrete or mixed parametrizations, or every
Nevergrad executor/constraint combination. Nevergrad remains a runtime dependency
for parametrization, candidate, archive, callback, constraint, and executor
machinery.

## Install and run

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

The shared library is written to `dist/libmojo-nevergrad.so`. To use a separately
packaged build, set `MOJONEVERGRAD_LIB` to that library's path.

Manual ask/tell works exactly as in Nevergrad:

```python
import mojonevergrad as ng

optimizer = ng.optimizers.PSO(parametrization=4, budget=100)
for _ in range(optimizer.budget):
    candidate = optimizer.ask()
    loss = float((candidate.value**2).sum())
    optimizer.tell(candidate, loss)

best = optimizer.provide_recommendation()
print(best.value)
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
6.8.0-136-generic, Python 3.13.14, Nevergrad 1.0.12:

| case | mojo-nevergrad | nevergrad | result |
| --- | ---: | ---: | ---: |
| OnePlusOne.ask, 1,000,000d | 38.742 ms | 39.714 ms | 1.03x faster |
| DE.ask, 100,000d | 3.958 ms | 285.446 ms | 72.11x faster |
| TwoPointsDE.ask, 100,000d | 4.118 ms | 4.547 ms | 1.10x faster |
| PSO.ask, 100,000d | 6.904 ms | 10.698 ms | 1.55x faster |
| DE.minimize sphere, 64d/300 evals | 134.017 ms | 178.481 ms | 1.33x faster |

The large DE gain comes from replacing Nevergrad's Python per-coordinate
binomial-crossover loop with one fused pass. Two-point DE is a modest win because
upstream already uses vectorized NumPy slicing; its Mojo update now processes the
copy and mutation ranges with the CPU's native `float64` SIMD width and scalar
tails. PSO uses the same SIMD-width strategy. `OnePlusOne` scales the NumPy-owned
random buffer in place, removing an allocation and copy while keeping the FFI
boundary zero-copy.

No GPU or multithreaded kernel path is included. The update kernels perform only
about 0.06 to 0.2 floating-point operations per byte moved, far below the roughly
2 flop/byte threshold where transfer and launch overhead can be justified. The
million-element scaling pass itself takes less than a millisecond, so CPU thread
launch and synchronization overhead are not warranted either.

Objective-function time is not included in the `ask` rows. For expensive real-world
objectives, optimizer-update time is usually a small part of the total run.

## How it works

Nevergrad owns optimizer control flow and candidate semantics. Candidate vectors are
converted to contiguous `float64` standardized coordinates. Before every FFI call,
Python validates dtype, contiguity, equal buffer lengths, and crossover indices,
keeps all NumPy owners alive for the duration of the call, and derives the element
count from the validated arrays. Empty arrays do not cross the boundary. Python then
passes buffer addresses and scalar settings through `ctypes`; Mojo reconstructs
mutable `UnsafePointer` values inside each exported function.

DE donor creation and crossover happen in one row-major pass, avoiding temporary
arrays and Python coordinate loops. PSO fuses velocity, cognitive and social terms,
position update, and clipping into one pass. Buffers are allocated and owned by
NumPy, so no allocation or ownership crosses the FFI boundary.

MIT licensed.
