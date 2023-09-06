from random import Random
from genetic_cnn_optimizer import CNNChromo


class GeneticCNN_finder:
    def __init__(self, max_gen, cross_prob, mutation_prob, max_population,
                 survive_percent, random_state, accuracy_function):
        self.max_gen = max_gen
        self.cross_prob = cross_prob
        self.mutation_prob = mutation_prob
        self.max_population = max_population
        self.survive_percent = survive_percent
        self.random = Random(random_state)
        self.accuracy_function = accuracy_function

    def selection(self, population):  # Roulette wheel selection method
        population_fitness = [p.fitness for p in population]
        total = sum(population_fitness)
        percentage = [round((x / total) * 100) for x in population_fitness]
        selection_wheel = []
        for pop_index, num in enumerate(percentage):
            selection_wheel.extend([pop_index] * num)
        parent1_ind = self.random.choice(selection_wheel)
        parent2_ind = self.random.choice(selection_wheel)
        return population[parent1_ind], population[parent2_ind]

    def run(self):
        generation = 0
        population = [CNNChromo(self.random) for _ in range(self.max_population)]
        elit = None

        while generation < self.max_gen:
            # ---------- obtain validation accuracy
            for p in population:
                p.accuracy = self.accuracy_function(p.chromo)

            # ---------- calculate each individual fitness
            if generation < self.max_gen / 2:
                sum_accuracies = sum([p.accuracy for p in population])
                for p in population:
                    p.fitness = p.accuracy / sum_accuracies
            else:
                population.sort(key=lambda p: p.accuracy, reverse=True)
                n = len(population)
                sum_ranks = (n * (n + 1)) / 2
                for i in range(n):
                    p = population[i]
                    r = i + 1
                    p.fitness = (n + 1 - r) / sum_ranks

            # ---------- get individual with best fitness
            elit = population[0]
            for p in population:
                if p.fitness > elit.fitness:
                    elit = p

            children_list = []
            next_population = []

            # ---------- crossover
            for _ in range(self.max_population):
                if self.random.random() <= self.cross_prob:
                    parent1, parent2 = self.selection(population)
                    if (generation / self.max_gen) < self.random.random():
                        children_list.extend(parent1.crossover_sequential(parent2, self.random))
                    else:
                        children_list.extend(parent1.crossover_binary(parent2, self.random))

            # ---------- mutation
            if generation < self.max_gen / 2:
                next_population = children_list
            else:
                survivors = population[:round(len(population) * self.survive_percent)]
                next_population.extend(survivors)
                next_population.extend(children_list)
                next_population.append(elit)

            for i in range(len(next_population)):
                if self.random.random() < self.mutation_prob:
                    next_population[i] = next_population[i].mutate(self.random)

            # ---------- fit size of next_population
            diff_n = len(next_population) - self.max_population
            if diff_n < 0:
                for _ in range(-diff_n):
                    next_population.append(CNNChromo(self.random))
            elif diff_n > 0:
                for _ in range(diff_n):
                    rnd_index = self.random.randint(0, len(next_population) - 1)
                    next_population.pop(rnd_index)

            # ---------- ensure existence of elit
            if elit not in next_population:
                rnd_index = self.random.randint(0, len(next_population) - 1)
                next_population[rnd_index] = elit

            population = next_population
            generation += 1

        return elit

