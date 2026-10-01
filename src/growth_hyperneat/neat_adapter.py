"""Thin adapter: CPPN topology, mutation, crossover and species stay in NEAT."""

from importlib.metadata import version
from pathlib import Path

import neat
import numpy as np

from .expression import ESDeveloper
from .model import Substrate
from .runtime import GrowingRNN


def load_neat_config(path=None, dimensions=3, population_size=12):
    """Load a standard DefaultGenome config; bundled profile uses 2*d+2 inputs.

    The bundled profile is tested with neat-python 0.92, matching the host evo
    application. Custom configurations may use a custom feature encoder instead.
    No custom reproduction/speciation or embodied composite genome is required.
    """
    if version("neat-python") != "0.92":
        raise RuntimeError("This adapter is tested with neat-python==0.92")
    if dimensions not in (2, 3) or type(population_size) is not int or population_size < 2:
        raise ValueError("dimensions must be 2/3 and population_size at least 2")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(path or Path(__file__).with_name("default_neat.ini")),
    )
    if path is None:
        config.genome_config.num_inputs = 2 * dimensions + 2
        config.genome_config.input_keys = [-i - 1 for i in range(2 * dimensions + 2)]
        config.pop_size = population_size
    if not config.genome_config.feed_forward:
        raise ValueError("CPPN must be feed-forward/stateless; the expressed RNN is separate")
    return config


class NeatCPPN:
    """Reference NEAT activations over independent queries, with arbitrary batch axes.

    Uses the standard scalar phenotype, so all native activation/aggregation
    functions work. Construct a new adapter after mutating its source genome.
    """

    def __init__(self, genome, config):
        if not config.genome_config.feed_forward:
            raise ValueError("CPPN must be feed-forward")
        self.network = neat.nn.FeedForwardNetwork.create(genome, config)
        self.inputs = config.genome_config.num_inputs
        self.outputs = config.genome_config.num_outputs

    def query(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim == 0 or values.shape[-1] != self.inputs:
            raise ValueError("incorrect CPPN input shape")
        if not np.isfinite(values).all():
            raise ValueError("CPPN inputs must be finite")
        batch = values.reshape(-1, self.inputs)
        result = np.empty((len(batch), self.outputs))
        for index, row in enumerate(batch):
            result[index] = self.network.activate(row.tolist())
        if not np.isfinite(result).all():
            raise ValueError("CPPN output must be finite")
        return result.reshape(values.shape[:-1] + (self.outputs,))


def express_genome(genome, config, substrate: Substrate, developer=None, **runtime_options):
    """Develop a native DefaultGenome into a new RNN for a fitness evaluation."""
    developer = developer or ESDeveloper()
    result = developer.develop(NeatCPPN(genome, config), substrate)
    return GrowingRNN(result.network, **runtime_options)
