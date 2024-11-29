import random
from Chromosome import Chromosome
from Population import Population


class GeneticAlgorithm:
    @staticmethod
    def select_Wheel(pop):
        partial_sum = 0
        sum_fitness = sum(chromosome.fitness for chromosome in pop.get_chromosomes())

        # Generate a random selection threshold
        random_shot = random.random() * sum_fitness

        # Find the chromosome corresponding to the random selection
        for chromosome in pop.get_chromosomes():
            partial_sum += chromosome.fitness
            if partial_sum >= random_shot:
                return chromosome

        # In case of rounding issues, return the last chromosome
        return pop.get_chromosomes()[-1]

    @staticmethod
    def crossover_chromosomes(parent1, parent2):
        if random.random() < 0.7:  # Default crossing rate
            # Create children
            child1 = Chromosome(len(parent1.get_genes()))
            child2 = Chromosome(len(parent2.get_genes()))

            # Perform one-point crossover
            crossover_point = random.randint(1, len(parent1.get_genes()) - 1)
            child1.genes = parent1.get_genes()[:crossover_point] + parent2.get_genes()[crossover_point:]
            child2.genes = parent2.get_genes()[:crossover_point] + parent1.get_genes()[crossover_point:]

            return child1, child2
        else:
            # If no crossover, return parents as offspring
            return parent1, parent2

    @staticmethod
    def mutate_chromosome(chromosome, mutation_rate=0.1):
        if random.random() < mutation_rate:
            # Select a random bit position to mutate
            random_bit_position = random.randint(0, len(chromosome.get_genes()) - 1)

            # Flip the bit at the chosen position
            chromosome.genes[random_bit_position] = 1 - chromosome.genes[random_bit_position]

        return chromosome

    @staticmethod
    def evolve(pop, num_elite_chromosomes=1, mutation_rate=0.1, neural_net=None, train_data=None, val_data=None, epochs=5, learning_rate=0.001):
        # Create a new population object
        new_pop = Population(size=0,  # We'll manually add chromosomes
                            num_genes=len(pop.get_chromosomes()[0].get_genes()),
                            neural_net=neural_net,  # Pass the neural network for recalculation
                            train_data=train_data,
                            val_data=val_data,
                            epochs=epochs,
                            learning_rate=learning_rate)

        # Retain the elite chromosomes
        for i in range(num_elite_chromosomes):
            new_pop.get_chromosomes().append(pop.get_chromosomes()[i])

        # Generate new chromosomes through crossover and mutation
        while len(new_pop.get_chromosomes()) < len(pop.get_chromosomes()):
            # Select parents using roulette-wheel selection
            parent1 = GeneticAlgorithm.select_Wheel(pop)
            parent2 = GeneticAlgorithm.select_Wheel(pop)

            # Perform crossover
            child1, child2 = GeneticAlgorithm.crossover_chromosomes(parent1, parent2)

            # Apply mutation
            GeneticAlgorithm.mutate_chromosome(child1, mutation_rate)
            GeneticAlgorithm.mutate_chromosome(child2, mutation_rate)

            # Add children to the new population
            new_pop.get_chromosomes().append(child1)

            # Ensure the second child does not exceed population size
            if len(new_pop.get_chromosomes()) < len(pop.get_chromosomes()):
                new_pop.get_chromosomes().append(child2)

        # Recalculate fitness for all chromosomes in the new population
        for chromosome in new_pop.get_chromosomes():
            # Use the same neural network and freeze layers based on the chromosome's genes
            neural_net.freeze_layers(chromosome.get_genes())
            chromosome.evaluate_fitness(neural_net, train_data, val_data, epochs=epochs, learning_rate=learning_rate)

        # Sort the new population by fitness
        new_pop.get_chromosomes().sort(key=lambda x: x.fitness, reverse=True)

        return new_pop



