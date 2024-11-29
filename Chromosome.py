import random
import numpy as np
from NeuralNet import NeuralNet

class Chromosome:
    def __init__(self, num_genes):
        self.genes = [random.randint(0, 1) for _ in range(num_genes)]  # Randomly initialize genes
        self.fitness = 0.0

    def get_genes(self):
        """Return the genes."""
        return self.genes

    def evaluate_fitness(self, neural_net, train_data, val_data, epochs=5, learning_rate=0.001):
        # Apply the gene configuration to freeze layers
        neural_net.freeze_layers(self.genes)

        # Train the neural network
        history = neural_net.train_model(train_data, val_data, epochs=epochs, learning_rate=learning_rate)

        # Use the final validation accuracy as the fitness score
        self.fitness = history['accuracy'][-1]
        return self.fitness

    def __str__(self):
        return f"Genes: {self.genes}, Fitness: {self.fitness:.4f}"
    
if __name__ == "__main__":
    # Parameters
    INPUT_SIZE = 10
    HIDDEN_SIZES = [128, 64, 32, 16, 8]
    OUTPUT_SIZE = 3
    NB_GENES = len(HIDDEN_SIZES)  # Each hidden layer is a gene
    EPOCHS = 5
    POPULATION_SIZE = 10

    # Generate mock data
    (X_train, y_train), (X_val, y_val) = NeuralNet.generate_mock_data(INPUT_SIZE, OUTPUT_SIZE)
    train_data = (X_train, y_train)
    val_data = (X_val, y_val)

    # Initialize population
    population = [Chromosome(NB_GENES) for _ in range(POPULATION_SIZE)]

    # Evaluate fitness of each chromosome
    for i, chromosome in enumerate(population):
        print(f"Evaluating Chromosome {i + 1}/{POPULATION_SIZE}...")
        net = NeuralNet(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
        fitness = chromosome.evaluate_fitness(net, train_data, val_data, epochs=EPOCHS)
        print(chromosome)

    # Sort chromosomes by fitness (higher is better)
    population.sort(key=lambda chrom: chrom.fitness, reverse=True)
    print("\nTop Chromosome:")
    print(population[0])