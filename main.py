import pandas as pd
import numpy as np
import random
import string

class EvolutionaryOptimizer:
    def __init__(self, dataframe, population_size, max_cost, mutation_rate):
        self.dataframe = dataframe
        self.population_size = population_size
        self.max_cost = max_cost
        self.mutation_rate = mutation_rate

        self.values = dataframe['value'].to_numpy()
        self.costs = dataframe['cost'].to_numpy()

        self.weighted_values = self.values / self.costs
        self.sorted_indices = np.argsort(-self.weighted_values)

        self.population, self.generation_numbers = self._generate_initial_population()
        self.best_solution = None
        self.best_fitness = -1

    def _is_within_tolerance(self, cost_sum):
        return cost_sum <= self.max_cost * 1.01

    def _generate_initial_population(self):
        population = []
        genomes_set = set()
        generation_numbers = []
        attempts = 0
        max_attempts = self.population_size * 100

        while len(population) < self.population_size and attempts < max_attempts:
            attempts += 1
            individual = np.zeros(len(self.values), dtype=bool)
            cost_sum = 0

            for index in self.sorted_indices:
                if cost_sum + self.costs[index] <= self.max_cost:
                    individual[index] = True
                    cost_sum += self.costs[index]

            num_tweaks = max(1, int(len(individual) * self.mutation_rate))
            for _ in range(num_tweaks):
                tweak_index = random.randrange(len(individual))
                if individual[tweak_index] or (not individual[tweak_index] and cost_sum + self.costs[tweak_index] <= self.max_cost):
                    cost_sum = cost_sum - self.costs[tweak_index] if individual[tweak_index] else cost_sum + self.costs[tweak_index]
                    individual[tweak_index] = not individual[tweak_index]

            individual_tuple = tuple(individual)
            if individual_tuple not in genomes_set and self._is_within_tolerance(cost_sum):
                population.append(individual)
                genomes_set.add(individual_tuple)
                generation_numbers.append(0)  # Initial generation number is 0

        return population, generation_numbers

    def _create_new_generation(self, parents):
        new_population = []
        new_generation_numbers = []
        genomes_set = set(map(tuple, self.population))

        for _ in range(self.population_size):
            parent1, parent2 = random.sample(parents, 2)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            child_cost = np.sum(self.costs[child])

            child_tuple = tuple(child)
            if child_tuple not in genomes_set and self._is_within_tolerance(child_cost):
                new_population.append(child)

                # Find indices of parents in the population
                index_parent1 = self._find_parent_index(parent1)
                index_parent2 = self._find_parent_index(parent2)

                parent_gen = max(self.generation_numbers[index_parent1],
                                 self.generation_numbers[index_parent2])
                new_generation_numbers.append(parent_gen + 1)
                genomes_set.add(child_tuple)

        return new_population, new_generation_numbers

    def _find_parent_index(self, parent):
        for i, individual in enumerate(self.population):
            if np.array_equal(individual, parent):
                return i
        return -1  # In case the parent is not found, though this should not happen


    def _fitness(self, individual):
        return np.sum(self.values[individual]) if np.sum(self.costs[individual]) <= self.max_cost else 0

    def _select_parents(self):
        fitness_scores = [self._fitness(individual) for individual in self.population]
        sorted_population = [x for _, x in sorted(zip(fitness_scores, self.population), key=lambda pair: pair[0], reverse=True)]
        return sorted_population[:2]

    def _crossover(self, parent1, parent2):
        return np.array([random.choice(pair) for pair in zip(parent1, parent2)])

    def _mutate(self, individual):
        mutation_indices = np.random.rand(len(individual)) < self.mutation_rate
        individual[mutation_indices] = ~individual[mutation_indices]
        return individual

    def _calculate_genome_stats(self, genome):
        fitness = self._fitness(genome)
        weight = np.sum(self.costs[genome])
        viability = 'Yes' if weight <= self.max_cost else 'No'
        return fitness, weight, viability

    def _calculate_weighted_value(self):
        weighted_values_df = pd.DataFrame({
            'object': self.dataframe['object'],
            'value': self.values,
            'cost': self.costs,
            'weighted_value': self.weighted_values
        })
        sorted_weighted_values_df = weighted_values_df.loc[self.sorted_indices].reset_index(drop=True)
        return sorted_weighted_values_df

    def verbose(self, generation, individual, fitness):
        print(f"Generation {generation}: Best fitness = {fitness}")
        print(f"Individual: {'-'.join(['1' if gene else '0' for gene in individual])}\n")

    def export_genomes_to_excel(self, file_name='genomes.xlsx'):
        with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
            object_names = self.dataframe['object'].tolist()
            gene_column_headers = {i: object_names[i] for i in range(len(object_names))}

            genome_repository_data = []
            for i, genome in enumerate(self.population):
                fitness, weight, viability = self._calculate_genome_stats(genome)
                genome_data = {
                    **{gene_column_headers[j]: gene for j, gene in enumerate(genome)},
                    'Fitness': fitness,
                    'Weight': weight,
                    'Viable': viability,
                    'Generation': self.generation_numbers[i]
                }
                genome_repository_data.append(genome_data)

            genomes_df = pd.DataFrame(genome_repository_data)
            genomes_df.index = [f'Galapagos ID #{i + 1}' for i in range(len(self.population))]
            genomes_df.to_excel(writer, sheet_name='Genome Repository', index_label='Galapagos ID')

            sorted_input_df = self._calculate_weighted_value()
            sorted_input_df.to_excel(writer, sheet_name='Input Parameters', index=False)

        print(f'Exported the repository of genomes to {file_name}')

    def run(self, generations):
        print("Starting the optimization process...")
        for generation in range(generations):
            parents = self._select_parents()
            current_best_fitness = self._fitness(parents[0])
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = parents[0]
                self.verbose(generation, self.best_solution, self.best_fitness)
            new_population, new_generation_numbers = self._create_new_generation(parents)
            self.population = new_population
            self.generation_numbers = new_generation_numbers

        self.dataframe = pd.DataFrame({
            'object': self.dataframe['object'],
            'active': self.best_solution
        })
        return self.dataframe

def create_random_objects(num_objects=1000, value_range=(100, 500), cost_range=(50, 200)):
    objects = [''.join(random.choices(string.ascii_uppercase + string.digits, k=5)) for _ in range(num_objects)]
    values = [random.randint(*value_range) for _ in range(num_objects)]
    costs = [random.randint(*cost_range) for _ in range(num_objects)]

    data = {
        'object': objects,
        'value': values,
        'cost': costs
    }

    return pd.DataFrame(data)

# Example usage
df = create_random_objects(num_objects=30)
optimizer = EvolutionaryOptimizer(df, population_size=200, max_cost=1500, mutation_rate=0.02)
optimized_df = optimizer.run(generations=100)
optimizer.export_genomes_to_excel('my_genomes.xlsx')
