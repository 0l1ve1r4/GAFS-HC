import random
from src.utils import Utils

class GeneticOperators:

    def __init__(self) -> None:
        self.utils = Utils()

    def create_population(self, population_size: int, len_attributes:int, last_generation_path = None):
        """Create the initial population with random genes (0 or 1)."""

        if last_generation_path != None:
            with open(last_generation_path, "r") as file:
                population = file.readlines()
            try:
                if population[0].strip() != "@FINISHED":
                    filtered_population = []
                    for chromosome in population[1:]:
                        filtered_population.append([int(gene) for gene in chromosome.strip().split(",")])
                    generation = int(population[0].strip()) 
                    
                    self.utils.debug(f"Generation {generation} Watered Population Discontinued", "warning")
                    self.utils.debug(f"Starting from {generation + 1}", "warning")
                    
                    return filtered_population, (generation + 1)
            except:
                pass

        population = []
        for _ in range(population_size):
            chromosome = [random.randint(0, 1) for _ in range(len_attributes)]

            if chromosome.count(1) == 0:
                chromosome[random.randint(0, len_attributes - 1)] = 1

            population.append(chromosome)

        return population, 0

    def roulette_selection(self, population, fitness_scores):
        """Selects individuals based on their fitness scores using the roulette selection method."""
        
        total_fitness = sum(fitness_scores)
        normalized_fitness = [score / total_fitness for score in fitness_scores]
        cumulative_probabilities = [sum(normalized_fitness[:i+1]) for i in range(len(normalized_fitness))]

        selected_population = []
        for _ in range(len(population)):
            rand_num = random.random()
            selected_index = next(i for i, cum_prob in enumerate(cumulative_probabilities) if rand_num <= cum_prob)
            selected_population.append(population[selected_index])

        return selected_population


    def tournament(self, population:list, fitness_scores:list, tournament_size = 2, k = 0.75) -> list:
        """Selects individuals based on their fitness scores using the tournament selection method.
            Also, applies elitism.
        """
        
        best_fitness = max(fitness_scores)
        best_fitness_index = fitness_scores.index(best_fitness)
        best_individual = population[best_fitness_index]
        
        selected_parents = []
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_candidates = [population[i] for i in tournament_indices]
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]

            if random.random() < k:
                selected_parent = tournament_candidates[tournament_fitness.index(max(tournament_fitness))]
            else:
                selected_parent = tournament_candidates[tournament_fitness.index(min(tournament_fitness))]

            selected_parents.append(selected_parent)
        
        random_index = random.randint(0, len(selected_parents) - 1)
        selected_parents[random_index] = best_individual # Elitism
            
        return selected_parents

    def pmx_crossover_chromossomes(self, parent1, parent2):
        """Performs the Partially Mapped Crossover (PMX) between two parents."""
        
        crossover_point1 = random.randint(0, len(parent1) - 1)
        crossover_point2 = random.randint(crossover_point1 + 1, len(parent1))

        child = [-1] * len(parent1)
        elements_in_child = set(parent1[crossover_point1:crossover_point2])

        for i in range(crossover_point1, crossover_point2):
            child[i] = parent1[i]

        for i in range(len(parent2)):
            if parent2[i] not in elements_in_child:
                empty_index = child.index(-1)
                child[empty_index] = parent2[i]
                elements_in_child.add(parent2[i])

        for i in range(len(child)):
            if child[i] == -1:
                child[i] = parent2[i] 

        return child

    def pmx_crossover(self, population:list, crossover_rate: float) -> list:
        """Performs the Partially Mapped Crossover (PMX) between pairs of parents."""
        
        new_population = []
        for i in range(0, len(population)-1, 2):
            parent1 = population[i]
            parent2 = population[i + 1]

            if random.random() < crossover_rate:
                child1 = self.pmx_crossover_chromossomes(parent1, parent2)
                child2 = self.pmx_crossover_chromossomes(parent2, parent1)
            else:
                child1 = parent1
                child2 = parent2

            new_population.append(child1)
            new_population.append(child2)

        return new_population
    
    def swap_mutation(self, population:list, mutation_rate: float) -> list:
        """Performs the Swap Mutation between pairs of parents."""
        
        for i in range(len(population)):
            if random.random() < mutation_rate:
                mutation_point1 = random.randint(0, len(population[i]) - 1)
                mutation_point2 = random.randint(0, len(population[i]) - 1)

                population[i][mutation_point1], population[i][mutation_point2] = population[i][mutation_point2], \
                                                                                 population[i][mutation_point1]

        return population
    
    def elitism(self, population:list, fitness_scores:list, num_elites:int) -> list:
        """Selects elite individuals based on fitness scores and replaces random individuals in the population with them."""
        
        elites = []
        for _ in range(num_elites):
            max_index, max_fitness = max(enumerate(fitness_scores), key=lambda x: x[1])
            elites.append(population[max_index])
            fitness_scores.pop(max_index)

        for elite_individual in elites:
            random_index = random.randint(0, len(population) - 1)
            population[random_index] = elite_individual

        return population