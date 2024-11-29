from Population import Population
from GeneticAlgorithm import GeneticAlgorithm
from NeuralNet import NeuralNet
from Chromosome import Chromosome

# Parameters
POPULATION_SIZE = 10
NB_GENES = 3  # Number of hidden layers, hence the number of genes
INPUT_SIZE = 10
HIDDEN_SIZES = [32, 16, 8]
OUTPUT_SIZE = 3
EPOCHS = 5
LEARNING_RATE = 0.001
MAX_GENERATIONS = 5  # Limit the number of generations
MUTATION_RATE = 0.1
NUM_ELITE_CHROMOSOMES = 1  # Number of elite chromosomes to keep per generation

# Main Program
if __name__ == "__main__":
    # Generate mock data
    (X_train, y_train), (X_val, y_val) = NeuralNet.generate_mock_data(INPUT_SIZE, OUTPUT_SIZE)
    train_data = (X_train, y_train)
    val_data = (X_val, y_val)

    # Initialize a neural network
    neural_net = NeuralNet(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

    # Initialize the first population
    generation_number = 0
    population = Population(POPULATION_SIZE, NB_GENES, neural_net, train_data, val_data, epochs=EPOCHS, learning_rate=LEARNING_RATE)

    # Print the initial population
    population.print_population(generation_number)

    # Evolve the population for a fixed number of generations
    for generation_number in range(1, MAX_GENERATIONS + 1):
        print(f"\n=== Generation {generation_number} ===")
        neural_net.reinitialize_weights()
        population = GeneticAlgorithm.evolve(
            pop=population,
            num_elite_chromosomes=NUM_ELITE_CHROMOSOMES,
            mutation_rate=MUTATION_RATE,
            neural_net=neural_net,  # Pass the same neural network instance
            train_data=train_data,
            val_data=val_data,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE
        )
        population.print_population(generation_number)


    # Print the final results
    print("\n=== Final Population ===")
    population.print_population(generation_number)
    print("\nBest Chromosome:")
    print(f"Genes: {population.get_chromosomes()[0].get_genes()} | Fitness: {population.get_chromosomes()[0].fitness:.4f}")
