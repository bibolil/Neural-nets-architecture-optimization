import random
from Chromosome import Chromosome
from NeuralNet import NeuralNet
class Population:
    def __init__(self, size, num_genes, neural_net, train_data, val_data, epochs=5, learning_rate=0.001):
        self.chromosomes = []
        for _ in range(size):
            chromosome = Chromosome(num_genes)
            chromosome.evaluate_fitness(neural_net, train_data, val_data, epochs, learning_rate)
            self.chromosomes.append(chromosome)

        # Sort chromosomes by fitness (higher fitness = better)
        self.chromosomes.sort(key=lambda x: x.fitness, reverse=True)

    def get_chromosomes(self):
        return self.chromosomes

    def print_population(self, gen_number):
        print("\n----------------------- Generation Summary -----------------------")
        print(f"Generation #{gen_number} | Fittest chromosome fitness: {self.get_chromosomes()[0].fitness:.4f}")
        print("------------------------------------------------------------------")
        for i, chromosome in enumerate(self.get_chromosomes()):
            print(f"Chromosome #{i + 1}: {chromosome.genes} | Fitness: {chromosome.fitness:.4f}")
    
if __name__ == "__main__":
    # Parameters
    POPULATION_SIZE = 10
    INPUT_SIZE = 10
    HIDDEN_SIZES = [32, 16, 8]
    OUTPUT_SIZE = 3
    EPOCHS = 5
    LEARNING_RATE = 0.001
    NB_GENES = len(HIDDEN_SIZES)

    # Generate mock data
    (X_train, y_train), (X_val, y_val) = NeuralNet.generate_mock_data(INPUT_SIZE, OUTPUT_SIZE)
    train_data = (X_train, y_train)
    val_data = (X_val, y_val)

    # Initialize a neural network
    neural_net = NeuralNet(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

    # Initialize population
    population = Population(POPULATION_SIZE, NB_GENES, neural_net, train_data, val_data, epochs=EPOCHS, learning_rate=LEARNING_RATE)

    # Perform roulette-wheel selection
    selected_chromosome = Population.select_Wheel(population)
    print(f"Selected Chromosome: {selected_chromosome.genes} | Fitness: {selected_chromosome.fitness:.4f}")

