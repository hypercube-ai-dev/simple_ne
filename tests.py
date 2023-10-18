from simple_ne.simple_ne import SimpleNEAgent, SimpleNENode, SimpleNEPopulation


def get_weights_tests():
    bp = SimpleNEPopulation(2, 2, 100, 10)
    test_genome = bp.create_genome()
    print(test_genome.get_weights())
    print(test_genome.get_weights(True))

get_weights_tests()