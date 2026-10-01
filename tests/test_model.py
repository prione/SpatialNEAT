import pytest

from spatial_neat import Edge, NetworkDefinition, Node, Region, Substrate


@pytest.mark.parametrize(
    "position, role, leak",
    [
        ((0,), "input", 1),
        ((0, 0, float("nan")), "input", 1),
        ((0, 0), "bad", 1),
        ((0, 0), "hidden", 0),
    ],
)
def test_invalid_node(position, role, leak):
    with pytest.raises(ValueError):
        Node("key", position, role, leak)


def test_substrate_and_definition_invariants():
    i, o = Node("i", (0, 0), "input"), Node("o", (1, 1), "output")
    with pytest.raises(ValueError, match="duplicate"):
        NetworkDefinition((i, i, o), ())
    with pytest.raises(ValueError, match="unknown"):
        NetworkDefinition((i, o), (Edge("missing", "o", 1),))
    with pytest.raises(ValueError, match="input nodes"):
        NetworkDefinition((i, o), (Edge("o", "i", 1),))
    with pytest.raises(ValueError):
        Region("bad", (1, 1), (0, 0))
    with pytest.raises(ValueError, match="dimensionality"):
        Substrate((i,), (o,), (Region("r", (-1, -1, -1), (1, 1, 1)),))
