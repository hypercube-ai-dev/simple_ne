"""
Very basic neat-python training script for .
"""
import gymnasium as gym
import os
import neat


def play_game(net):
    env = gym.make("CarRacing-v2")
    return 0.0

def eval_genomes_parallel(genomes, config, callback):
    raise NotImplementedError

def eval_genomes(genomes, config):
    fitness_list = []
    for g in genomes:
        reward = play_game(g)

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    p = neat.Checkpointer.restore_checkpoint('neat-car-racing')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, './configs/neat_base.cfg')
    run(config_path)