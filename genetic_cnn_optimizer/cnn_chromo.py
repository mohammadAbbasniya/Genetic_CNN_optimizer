from random import Random

POSSIBLE_LAYERS_N = [3, 5, 7]
POSSIBLE_FILTERS = [16, 32, 64, 128, 256]
POSSIBLE_KERNELS = [3, 5, 7]
N_LAYERS_MUTATE_PROBA = 0.3  # probability of changing number of layers of a chromosome


class CNNChromo:
    def __init__(self, random: 'Random', chromo: list = None):
        if chromo is None:
            n = random.choice(POSSIBLE_LAYERS_N)
            self.chromo = [n]
            for i in range(n):
                self.chromo.extend([
                    random.choice(POSSIBLE_FILTERS),  # L_i
                    random.choice(POSSIBLE_KERNELS),  # K_i
                ])
        else:
            self.chromo = chromo.copy()

        self.accuracy = None
        self.fitness = None

    def copy(self):
        copy_cnn_chromo = CNNChromo(Random(), chromo=self.chromo.copy())
        copy_cnn_chromo.fitness = self.fitness
        copy_cnn_chromo.accuracy = self.accuracy
        return copy_cnn_chromo

    def crossover_sequential(self, other_chromo: 'CNNChromo', random: 'Random'):
        n1 = self.chromo[0]
        chromo1 = self.chromo.copy()

        n2 = other_chromo.chromo[0]
        chromo2 = other_chromo.chromo.copy()

        pivot = random.randint(1, min(n1, n2) * 2 - 1)
        ofsp1 = chromo2[:pivot + 1]
        ofsp1.extend(chromo1[pivot + 1:])

        ofsp2 = chromo1[:pivot + 1]
        ofsp2.extend(chromo2[pivot + 1:])

        ofsp1[0] = len(ofsp1) // 2
        ofsp2[0] = len(ofsp2) // 2

        return CNNChromo(random, ofsp1), CNNChromo(random, ofsp2)

    def crossover_binary(self, other_chromo: 'CNNChromo', random: 'Random'):
        n1 = self.chromo[0]
        ofsp1 = self.chromo.copy()

        n2 = other_chromo.chromo[0]
        ofsp2 = other_chromo.chromo.copy()

        binary_list_len = max(n1, n2) * 2
        binary_list = [random.randint(0, 1) for _ in range(binary_list_len + 1)]

        for index in range(1, binary_list_len):
            if index < len(ofsp1) and index < len(ofsp2):
                if binary_list[index] == 1:
                    ofsp1[index] = other_chromo.chromo[index]
                    ofsp2[index] = self.chromo[index]

        return CNNChromo(random, ofsp1), CNNChromo(random, ofsp2)

    def mutate(self, random: 'Random'):
        mutated_chromo = self.chromo.copy()

        if random.random() < N_LAYERS_MUTATE_PROBA:  # change number of layers
            rand_n = random.choice(POSSIBLE_LAYERS_N)
            if rand_n < mutated_chromo[0]:  # decrease layers number
                mutated_chromo = mutated_chromo[: 1 + rand_n * 2]
                mutated_chromo[0] = rand_n
            else:  # increase layers number
                for i in range(rand_n - mutated_chromo[0]):
                    mutated_chromo.extend([
                        random.choice(POSSIBLE_FILTERS),  # L_i
                        random.choice(POSSIBLE_KERNELS),  # K_i
                    ])
                mutated_chromo[0] = rand_n

        else:  # change a single parameter
            rand_index = random.randint(1, mutated_chromo[0] * 2)
            if rand_index % 2 == 0:  # its index of a kernel_size
                mutated_chromo[rand_index] = random.choice(POSSIBLE_KERNELS)
            else:  # its index of a filters_number
                mutated_chromo[rand_index] = random.choice(POSSIBLE_FILTERS)

        return CNNChromo(random, mutated_chromo)

    def __str__(self):
        return f'chromosome: {self.chromo}   accuracy: {self.accuracy * 100:.2f}%'