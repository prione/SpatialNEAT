"""Compare array execution with an independently installed reference package.

Run with this checkout's src on PYTHONPATH and --reference-package pointing at
an older installed spatial_neat directory. Reference code is never vendored.
Initialization/restore time is excluded; every timed trial starts at the same
snapshot. The script checks states, outputs and gates before measuring medians.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from spatial_neat import Development, Edge, GrowingRNN, NetworkDefinition, Node, __version__


def load_reference(directory):
    name = "_spatial_neat_reference"
    directory = Path(directory).resolve()
    spec = importlib.util.spec_from_file_location(
        name, directory / "__init__.py", submodule_search_locations=[str(directory)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sample(hidden, edges, seed=7, grown=False):
    rng = np.random.default_rng(seed)
    nodes = [Node(f"i{i}", (0, 0, -1), "input") for i in range(8)]
    nodes += [Node(f"h{i}", (0, 0, 0), "hidden", leak=0.3) for i in range(hidden)]
    nodes += [Node(f"o{i}", (0, 0, 1), "output") for i in range(4)]
    keys = [(a.key, b.key) for a in nodes for b in nodes if b.role != "input"]
    chosen = rng.choice(len(keys), size=min(edges, len(keys)), replace=False)
    weights = tuple(Edge(*keys[i], float(rng.uniform(-1, 1))) for i in chosen)
    brain = GrowingRNN(NetworkDefinition(tuple(nodes), weights))
    brain.step(rng.uniform(-1, 1, 8))
    for edge in weights[:16]:
        brain.set_delta(*edge.key, float(rng.uniform(-0.1, 0.1)))
    if grown:
        Development(brain).grow(
            NetworkDefinition(
                tuple(nodes) + (Node("new", (1, 0, 0), "hidden", leak=0.2),),
                (Edge("i0", "new", 0.5), Edge("new", "o0", 0.3)),
            )
        )
    return brain.snapshot()


def measure(snapshot, reference_class, steps=256, repeats=5):
    values = np.random.default_rng(13).uniform(
        -1, 1, (steps, len([n for n in snapshot["network"]["nodes"] if n["role"] == "input"]))
    )
    old, new = reference_class.from_snapshot(snapshot), GrowingRNN.from_snapshot(snapshot)
    error = 0.0
    for inputs in values:
        a, b = old.step(inputs), new.step(inputs)
        np.testing.assert_allclose([b[k] for k in a], list(a.values()), atol=1e-12, rtol=1e-12)
        before, after = old.state, new.state
        difference = np.abs(np.array([before[k] for k in before]) - [after[k] for k in before])
        error = max(error, float(difference.max()))
        np.testing.assert_allclose(
            list(before.values()), [after[k] for k in before], atol=1e-12, rtol=1e-12
        )
        assert old.gates == new.gates
    times = {"reference": [], "array": []}
    for repeat in range(repeats):
        order = ("reference", "array") if repeat % 2 == 0 else ("array", "reference")
        for label in order:
            brain = (reference_class if label == "reference" else GrowingRNN).from_snapshot(
                snapshot
            )
            start = perf_counter()
            for inputs in values:
                brain.step(inputs)
            times[label].append(perf_counter() - start)
    old_time, new_time = (float(np.median(times[k])) for k in ("reference", "array"))
    return {
        "nodes": len(snapshot["network"]["nodes"]),
        "edges": len(snapshot["weights"]),
        "reference_seconds": old_time,
        "array_seconds": new_time,
        "isolated_speedup": old_time / new_time,
        "max_state_error": error,
        "trials_seconds": times,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-package", required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if min(args.steps, args.repeats) < 1:
        parser.error("steps and repeats must be positive")
    reference = load_reference(args.reference_package)
    report = {
        "current": __version__,
        "reference": reference.__version__,
        "scope": "isolated fixed-weight runtime, not end-to-end application speedup",
        "steps": args.steps,
        "repeats": args.repeats,
        "cases": {},
    }
    for label, hidden, edges, grown in (
        ("small", 16, 200, False),
        ("medium", 80, 3000, False),
        ("large", 240, 20000, False),
        ("just_grown", 80, 3000, True),
    ):
        case = measure(
            sample(hidden, edges, grown=grown), reference.GrowingRNN, args.steps, args.repeats
        )
        report["cases"][label] = case
        print(
            f"{label}: {case['nodes']} nodes / {case['edges']} edges; "
            f"{case['isolated_speedup']:.2f}x; state error {case['max_state_error']:.3g}"
        )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
