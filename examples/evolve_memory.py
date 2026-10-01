"""Small native-NEAT smoke example; not a performance benchmark or solved-task claim."""

import argparse
import json
import random

import neat

from growth_hyperneat import (
    ESConfig,
    ESDeveloper,
    ExpressionLimitError,
    Node,
    Region,
    Substrate,
    express_genome,
    load_neat_config,
)


def run(generations=3, population_size=8, seed=7):
    config = load_neat_config(population_size=population_size)
    substrate = Substrate(
        (Node("signal", (0, 0, -0.75), "input"),),
        (Node("prediction", (0, 0, 0.75), "output"),),
        (Region("core", (-1, -1, -1), (1, 1, 1), leak=0.3),),
    )
    developer = ESDeveloper(
        ESConfig(
            max_depth=2, direct_links=True, iterations=1, max_hidden_nodes=256, max_queries=100000
        )
    )
    evaluations = []

    def evaluate(genomes, native_config):
        for key, genome in genomes:
            try:
                brain = express_genome(genome, native_config, substrate, developer)
                loss = 0.0
                # Recall the sign of a one-step cue during a short zero-input delay.
                for sign in (-1, 1):
                    brain.reset()
                    brain.step([sign])
                    for _ in range(4):
                        prediction = brain.step([0])["prediction"]
                        loss += (prediction - sign) ** 2
                complexity = brain.complexity
                genome.fitness = -loss / 8 - 0.001 * complexity["hidden"]
            except ExpressionLimitError:
                genome.fitness = -1000.0
            evaluations.append(key)

    previous_random = random.getstate()
    try:
        random.seed(seed)
        population = neat.Population(config)
        winner = population.run(evaluate, generations)
        return {
            "generations": generations,
            "evaluations": len(evaluations),
            "best_fitness": winner.fitness,
            "genome_class": type(winner).__name__,
        }
    finally:
        random.setstate(previous_random)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.generations < 1:
        parser.error("generations must be positive")
    print(json.dumps(run(args.generations, args.population, args.seed)))


if __name__ == "__main__":
    main()
