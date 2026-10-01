# SpatialNEAT

SpatialNEAT is a Python library for turning spatial connection patterns into
recurrent neural networks. It combines standard NEAT evolution with 2D/3D
ES-HyperNEAT: a coordinate-based network generator discovers hidden neurons and
connections within user-defined regions.

Structural development is optional. Networks can run with a fixed structure, or
an application can explicitly grow, prune, reposition, reconfigure, and
approximately merge neurons during execution.

## Features

- Adaptive hidden-neuron placement using quadtrees in 2D and octrees in 3D.
- Multiple spatial regions, connection filters, and explicit exploration budgets.
- Standard `neat-python` genomes, mutation, crossover, reproduction, and speciation.
- Sparse, synchronous recurrent execution with self-connections and cycles.
- Optional structural changes that preserve retained state and learned weight offsets.
- Isolated change previews, validated commits, rollback, and JSON snapshots.

## Installation

Requires Python 3.10 or later, NumPy, and `neat-python==0.92`.
The NEAT adapter is currently tested against that exact version.

Install from the repository:

```sh
git clone https://github.com/prione/SpatialNEAT.git
cd SpatialNEAT
python -m pip install .
```

The distribution name is `spatial-neat`; the Python import name is `spatial_neat`.
The package is not currently published on PyPI.

For tests and examples:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/growth_demo.py
python examples/development_demo.py
python examples/evolve_memory.py --generations 3 --population 8
```

## Quick start

Input and output neurons are anchors chosen by the application. Hidden neurons
are discovered inside one or more regions. This example expresses a randomly
initialized standard NEAT genome as a 3D recurrent network:

```python
import neat

from spatial_neat import (
    ESConfig,
    ESDeveloper,
    GrowingRNN,
    NeatCPPN,
    Node,
    Region,
    Substrate,
    load_neat_config,
)

config = load_neat_config(dimensions=3, population_size=12)
genome = neat.DefaultGenome(1)
genome.configure_new(config.genome_config)
cppn = NeatCPPN(genome, config)

substrate = Substrate(
    inputs=(Node("sensor", (0, 0, -1), "input"),),
    outputs=(Node("action", (0, 0, 1), "output"),),
    regions=(Region("core", (-1, -1, -1), (1, 1, 1), leak=0.2),),
)
developer = ESDeveloper(ESConfig(max_depth=2))
result = developer.develop(cppn, substrate)
brain = GrowingRNN(result.network)

for _ in range(4):
    action = brain.step({"sensor": 0.5})["action"]
```

Random initialization does not produce a trained controller. A constant or
unhelpful connection pattern may yield no hidden neurons or connections.

Use `neat.Population(config).run(evaluate_genomes, generations)` for evolution.
Inside the evaluator, `express_genome(genome, config, substrate, developer)`
creates a fresh RNN. Fitness, tasks, and evaluation are application-defined.

For 2D, use two-component coordinates and `load_neat_config(dimensions=2)`.

## Spatial expression

A compositional pattern-producing network (CPPN) maps source and target
coordinates to connection values. The default feature vector contains source
coordinates, target coordinates, distance, and a constant 1: six inputs in 2D
and eight in 3D. The first CPPN output is clipped to [-1, 1], then scaled by
`weight_scale`.

The developer adaptively subdivides regions, extracts connections using variance
and band thresholds, and iterates through discovered hidden neurons. It also
samples incoming patterns at output anchors. Hidden neurons and connections are
retained only when they lie on a structural path from an input to an output.

Coordinates are not normalized automatically; anchors and regions must share a
consistent coordinate system. Overlapping regions remain separate modules,
including when neurons occupy the same coordinates.

Useful configuration points:

- `initial_depth`, `max_depth`, variance thresholds, and band thresholds control
  sampling resolution. They do not guarantee an optimal layout or neuron count.
- `max_hidden_nodes` and `max_edges` limit candidates before reachability cleanup.
  `max_queries` limits unique CPPN queries per expression. Exceeding a budget
  raises `ExpressionLimitError` instead of returning a partial network.
- `direct_links=True` optionally adds direct input-to-output connections.
- `connection_filter(source_node, target_node)` restricts allowed connections.
- A custom stateless object with `query(values)` can replace `NeatCPPN`.
  `ESDeveloper(features=...)` supports custom coordinate feature encoders.

The extraction procedure follows the adaptive subdivision, variance, band
extraction, and network-completion ideas in
[Iterated ES-HyperNEAT](https://groups.csail.mit.edu/EVO-DesignOpt/gecco2011Proceedings/proceedings/p1539.pdf).
The library adds 3D regions, recurrent execution, resource budgets, and optional
structural development. It does not reproduce the paper's benchmark results.

## Recurrent execution and learning

Every non-input neuron, including outputs, updates synchronously from the
previous state:

```text
h_next = h + leak * (tanh(weighted_sum) - h)
```

An input → hidden → output path therefore needs at least two `step` calls.
Self-connections and cycles are supported. Supply a constant-valued input anchor
if the expressed network needs a bias signal.

By default, the sum of absolute recurrent weights into each neuron is limited
to 0.98. This conservative contraction constraint trades persistent memory for
stability. Set `recurrent_limit=None` to disable it and evaluate stability in
the application.

External learners can use `brain.set_delta(source_id, target_id, delta)` to set
a learned weight offset. The method returns the accepted offset after enforcing
the recurrent budget. The library does not provide reward learning, optimizers,
eligibility traces, or an automatic growth policy.

## Optional structural development

`Development` manages explicit structural changes. It is not required for
ordinary expression or recurrent execution.

For example, extend the search space while keeping existing region identities
and bounds:

```python
from spatial_neat import Development

grown_substrate = Substrate(
    inputs=substrate.inputs,
    outputs=substrate.outputs,
    regions=substrate.regions + (
        Region("extension", (-1, 1, -1), (1, 2, 1), leak=0.2),
    ),
)
candidate = developer.develop(cppn, grown_substrate)
development = Development(brain)
change = development.grow(candidate.network)

# New connections gradually become active.
development.mature(ticks=2)

# Reject the change and restore the previous RNN state.
change.rollback()
```

Available operations:

- `grow(candidate_network)` adds neurons and connections. Existing neurons,
  connections, states, weight offsets, and maturation progress are retained,
  even if they disappear from the candidate.
- `mature(ticks=...)` advances connection gates without advancing neural state.
  Normal `brain.step(...)` already advances maturation once per step.
- `prune(nodes=..., edges=...)` explicitly removes selected elements and
  incident connections.
- `move({node_id: coordinates})` changes positions while preserving computation.
  Passing `cppn=...` re-expresses weights of existing incident connections;
  it does not rediscover topology.
- `reconfigure(candidate_network, policy)` adopts the candidate topology
  exactly, including removals.
- `preview_merge(...)` and `merge(...)` approximately combine hidden neurons.

New neuron states start at zero. New connections mature from gate 0 to 1 over
eight steps by default. Under default maturation, additive growth preserves the
first post-growth output for retained outputs given the same inputs; subsequent
behavior is not guaranteed to remain unchanged. There is no incremental
expression cache: a new expression explores the supplied substrate again.

### Migration, previews, and rollback

`MigrationPolicy` controls retained weights, learned offsets, explicit state and
weight transfers, and permitted input/output changes. Reconfiguration rejects
port changes by default, changing the role of an existing neuron ID, and changes
between 2D and 3D. Additive growth permits new ports explicitly through its own
operation.

`require_output_paths=True` adds a structural reachability check. This check
includes zero-weight and immature connections; it does not prove functional
performance. Weight re-expression is immediate, unlike staged new connections.

```python
from spatial_neat import MigrationPolicy

prepared = development.preview(candidate.network, MigrationPolicy())
trial = prepared.trial()  # independent RNN for application-defined evaluation

change = development.commit(prepared)
change.rollback()
```

Commit rejects a preview if the original RNN's state, learned weights,
maturation, or structure changed after preparation. Trial activity and learning
are not automatically adopted.

Rollback restores the full RNN state, weights, learned offsets, and maturation
from before the latest structural change. It does not restore external
environments, random-number generators, or optimizer state. Only the latest
change handle can be rolled back, once.

### Approximate neuron merging

Merging is compression, not an exact equivalence transformation. It combines
hidden-neuron states and incoming weights by weighted averaging and sums outgoing
weights using a linear projection. Nonlinear activation and differing leak
values mean behavior can change.

When merged connections have different maturation gates, their current gates
are folded into the weights; their individual future maturation schedules are
lost. Select candidates and evaluate acceptable error or retraining in the
application, preferably through `preview_merge(...)` before committing.

## Saving and restoring

```python
import json

saved = json.dumps(brain.snapshot(), allow_nan=False)
restored = GrowingRNN.from_snapshot(json.loads(saved))
```

Snapshots contain the RNN, including structure, state, learned offsets, and
maturation. They do not contain the NEAT population, fitness history, external
learners, or random state. Use independent CPPN/RNN instances for parallel
evaluations.

## License

No license has been selected yet.
