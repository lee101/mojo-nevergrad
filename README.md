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
| OnePlusOne.ask, 1,000,000d | 44.680 ms | 37.387 ms | 1.20x slower |
| DE.ask, 100,000d | 5.052 ms | 355.279 ms | 70.33x faster |
| TwoPointsDE.ask, 100,000d | 6.430 ms | 7.031 ms | 1.09x faster |
| PSO.ask, 100,000d | 13.369 ms | 16.732 ms | 1.25x faster |
| DE.minimize sphere, 64d/300 evals | 124.618 ms | 166.298 ms | 1.33x faster |

The large DE gain comes from replacing Nevergrad's Python per-coordinate
binomial-crossover loop with one fused pass. Two-point DE is a modest win because
upstream already uses vectorized NumPy slicing. PSO's fused update uses the CPU's
native `float64` SIMD width with a scalar tail. `OnePlusOne` uses the Mojo scaling
kernel but is slightly slower than upstream in this measurement because allocation
and FFI overhead outweigh the simple compiled operation.

No GPU or multithreaded kernel path is included.

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
