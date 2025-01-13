import random
import math
import time
import pandas as pd


class City:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

    def distance(self, city):
        return math.sqrt((self.x - city.x) ** 2 + (self.y - city.y) ** 2)


# calculates the fitness of a path: higher fitness value means a better path
def _fitness(path):
    return 1 / _path_length(path)


# calculates the total distance of a given path
def _path_length(path):
    length = 0
    for i in range(len(path) - 1):
        length += path[i].distance(path[i + 1])

    return length


# creates a child path by combining two parent paths
def _crossover(parent1, parent2):
    start, end = sorted([random.randint(0, len(parent1) - 1) for _ in range(2)])
    child_p1 = parent1[start:end]
    child_p2 = [city for city in parent2 if city not in child_p1]
    child = child_p1 + child_p2
    return child


class GeneticAlgorithm:
    def __init__(self, cities, population_size=500, generations=1000, mutation_rate=0.02, elite_size=5):
        self.cities = cities
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population = self._initialize_population()

    # generates the initial population of paths
    def _initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = random.sample(self.cities, len(self.cities))
            population.append(individual)
        return population

    # selects paths from the current population for the next generation
    def _selection(self):
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, 5)
            selected.append(min(tournament, key=_path_length))
        return selected

    # makes random changes to a path to maintain diversity in the population
    def _mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                j = random.randint(0, len(individual) - 1)
                individual[i], individual[j] = individual[j], individual[i]

    # creates a new population by combining elite individuals and new offspring
    def _create_new_generation(self, selected_individuals):
        new_generation = []
        elite_individuals = sorted(self.population, key=_path_length)[:self.elite_size]
        new_generation.extend(elite_individuals)
        while len(new_generation) < self.population_size:
            parent1, parent2 = random.sample(selected_individuals, 2)
            child = _crossover(parent1, parent2)
            self._mutate(child)
            new_generation.append(child)
        return new_generation

    def run(self):
        best_overall_path = None
        best_overall_length = float('inf')
        for generation in range(self.generations):
            selected_individuals = self._selection()
            self.population = self._create_new_generation(selected_individuals)
            best_path_calculate = min(self.population, key=_path_length)
            best_length = _path_length(best_path_calculate)

            if best_length < best_overall_length:
                best_overall_path = best_path_calculate
                best_overall_length = best_length

            if (generation in [0] + list(range(self.generations // 10, self.generations, self.generations // 10))
                    + [self.generations - 1]):
                print(int(best_length))

        return best_overall_path


def load_predefined_data(dataset_name_load):
    try:
        xy_data_path = f'./data/{dataset_name_load}_xy.csv'
        name_data_path = f'./data/{dataset_name_load}_name.csv'

        xy_data = pd.read_csv(xy_data_path, header=None)
        name_data = pd.read_csv(name_data_path, header=None)

        return [City(name_data.iloc[i, 0], xy_data.iloc[i, 0], xy_data.iloc[i, 1]) for i in range(len(name_data))]
    except FileNotFoundError:
        print(f"Error: Dataset files '{dataset_name_load}_xy.csv' and/or "
              f"'{dataset_name_load}_name.csv' not found in the './data/' directory.")
        exit()


def generate_random_cities(n):
    return [City(f"City_{i}", random.uniform(-1000, 1000), random.uniform(-1000, 1000)) for i in range(n)]


if __name__ == "__main__":
    start_time = time.time()
    user_input = input("Enter the number of cities (N) or the dataset name (e.g., 'UK12'): ").strip()
    if user_input.isdigit():
        N = int(user_input)
        unprocessed_cities = generate_random_cities(N)
    else:
        dataset_name = user_input.lower()
        unprocessed_cities = load_predefined_data(dataset_name)

    ga = GeneticAlgorithm(unprocessed_cities, generations=200, population_size=1000, mutation_rate=0.02, elite_size=10)
    best_path = ga.run()

    print('')
    print(' -> '.join(city.name for city in best_path))
    print(int(_path_length(best_path)))

    end_time = time.time()
    execution_time = round(end_time - start_time, 4)
    print(execution_time)
