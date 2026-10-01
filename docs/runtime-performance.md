# Array-backed recurrent execution

SpatialNEAT 0.3.1 stores recurrent state, base weights, learned offsets and
maturation gates in float64 NumPy arrays. Immutable source/target indices,
input/output positions and recurrent row indices are compiled once per topology.
Effective gated weights are cached; offset changes refresh only the affected
connection, and maturation visits only connections whose gates are below one.

The public interface still uses stable string IDs and detached dictionaries.
Snapshots remain JSON dictionaries with schema version 1. Growth, pruning,
movement, merging, restoration and rollback rebuild or restore the compiled
layout by logical IDs, not previous array positions. Accumulation errors do not
advance state or maturation. The synchronous leaky-tanh update is unchanged.

Recurrent row validation now uses compiled row indices rather than repeatedly
scanning every connection for every dynamic neuron. It is O(nodes + edges).
Single-offset clipping examines only the relevant recurrent row. There is no
new reward rule, optimizer, GPU dependency or fixed recurrent architecture.

## Measurement

An independently installed 0.3.0 package is loaded under a separate namespace;
no old source implementation is copied into the library. Before timing, each
case checks outputs, all states and maturation gates over the same 256 inputs.
Five trials alternate reference/current order; initialization and restoration
are outside the timed segments. Both implementations construct public output
dictionaries and validate each input vector.

One Python 3.11 / NumPy 2.4.6 / WSL2 Linux container measurement:

| Synthetic graph | Nodes | Edges | Reference, 256 steps | Array runtime | Ratio |
|---|---:|---:|---:|---:|---:|
| Small | 28 | 200 | 19.33 ms | 1.64 ms | 11.8x |
| Medium | 92 | 3,000 | 265.28 ms | 2.69 ms | 98.6x |
| Large | 252 | 20,000 | 1,907.98 ms | 11.04 ms | 172.8x |
| Immediately after growth | 93 | 3,002 | 276.45 ms | 3.74 ms | 73.9x |

The maximum measured state difference was zero in these four cases. Randomized
2D/3D regression tests use numerical tolerances; bitwise equivalence across
platforms is not promised. Learned offsets, arbitrary node/edge ordering and
nonzero state are included in the comparisons. Existing lifecycle/migration
tests plus new array tests total 177 passing tests on both NumPy 1.26.4 and 2.4.6.

These are **isolated fixed-weight execution ratios**, not speedups for network
expression, evolution, training, topology changes or an entire application.
Hardware, load, graph size and the fraction of immature edges affect results.
Sparse storage remains O(nodes + edges); networks are not densified.

## Reproduce

Install a reference into a separate directory, and run this checkout from its root:

```sh
python -m pip install --no-deps --target /tmp/spatial-reference \
  'spatial-neat @ git+https://github.com/prione/SpatialNEAT.git@7ed3c6be0a63f4483336965ee12c4d7aeb7d89ab'
PYTHONPATH=src python benchmarks/runtime.py \
  --reference-package /tmp/spatial-reference/spatial_neat \
  --steps 256 --repeats 5 --output /tmp/spatial-runtime-benchmark.json
```

The reference needs the regular NumPy/NEAT dependencies in the active environment.
This command is a benchmark workflow, not a runtime switch. Ordinary applications
continue to call `GrowingRNN.step`, `set_delta`, `snapshot` and `Development`.
