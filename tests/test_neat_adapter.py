import random

import neat
import numpy as np
import pytest

from growth_hyperneat import (
    ESConfig,
    ESDeveloper,
    NeatCPPN,
    Node,
    Region,
    Substrate,
    express_genome,
    load_neat_config,
)


@pytest.mark.parametrize("dimensions", [2, 3])
def test_standard_neat_config_and_query_matches_native_network(dimensions):
    config = load_neat_config(dimensions=dimensions)
    genome = neat.DefaultGenome(1)
    genome.configure_new(config.genome_config)
    cppn = NeatCPPN(genome, config)
    queries = np.random.default_rng(5).normal(size=(2, 3, 2 * dimensions + 2))
    native = neat.nn.FeedForwardNetwork.create(genome, config)
    expected = [native.activate(row.tolist()) for row in queries.reshape(-1, cppn.inputs)]
    np.testing.assert_array_equal(cppn.query(queries).reshape(-1, 1), expected)
    np.testing.assert_array_equal(cppn.query(queries[::-1]), cppn.query(queries)[::-1])
    assert cppn.query(np.empty((0, cppn.inputs))).shape == (0, 1)
    with pytest.raises(ValueError):
        cppn.query(np.zeros(cppn.inputs + 1))
    with pytest.raises(ValueError):
        cppn.query(np.full(cppn.inputs, np.nan))


def test_default_genome_standard_population_two_generations_no_custom_neat_classes():
    config = load_neat_config(population_size=4)
    previous_random = random.getstate()
    try:
        random.seed(10)
        population = neat.Population(config)
        substrate = Substrate(
            (Node("sensor", (0, 0, -1), "input"),),
            (Node("action", (0, 0, 1), "output"),),
            (Region("core", (-1, -1, -1), (1, 1, 1)),),
        )
        developer = ESDeveloper(
            ESConfig(initial_depth=1, max_depth=1, iterations=0, direct_links=True)
        )
        evaluated = []

        def evaluate(genomes, native_config):
            assert native_config is config
            for key, genome in genomes:
                assert type(genome) is neat.DefaultGenome
                brain = express_genome(genome, config, substrate, developer)
                genome.fitness = -abs(brain.step([0.5])["action"] - 0.5)
                evaluated.append(key)

        winner = population.run(evaluate, 2)
        assert len(evaluated) == 8
        assert winner.fitness is not None
        assert type(population.reproduction) is neat.DefaultReproduction
        assert type(population.species) is neat.DefaultSpeciesSet
    finally:
        random.setstate(previous_random)


def test_cppn_must_be_stateless_even_when_controller_is_recurrent():
    config = load_neat_config()
    config.genome_config.feed_forward = False
    with pytest.raises(ValueError):
        NeatCPPN(neat.DefaultGenome(1), config)
