import numpy as np
import logging
from numpy.random import random
from numpy.random import choice

from constants import *

class Species(object):
    def __init__(self, genome=None):
        self.genomes = []
        if genome is not None:
            self.genomes.append(genome)

        self.staleness = 0
        self.average_fitness = 0.0
        self.top_fitness = 0.0
        self.max_fitness = 0.0

    def copy(self):
        result = Species()
        for genome in self.genomes:
            result.genomes.append(genome.copy())

        result.staleness = self.staleness
        result.average_fitness = self.average_fitness
        result.top_fitness = self.top_fitness
        result.max_fitness = self.max_fitness

        return result

    def add_genome(self, genome):
        self.genomes.append(genome)

    def same_species(self, genome):
        return self.genomes[0].same_species(genome)

    def cull_species(self, cut_to_one):
        if cut_to_one:
            remaining = 1
        else:
            remaining = (len(self.genomes)+1)/2

        self.genomes.sort(key = lambda x: -x.fitness)
        self.genomes = self.genomes[:remaining]

    def breed_child(self):
        if random() < CROSSOVER_CHANCE:
            g1 = self.genomes[choice(len(self.genomes))]
            g2 = self.genomes[choice(len(self.genomes))]
            child = g1.crossover(g2)
        else:
            child = self.genomes[choice(len(self.genomes))].copy()

        child.mutate()
        return child

    def calc_average_fitness(self):
        total = 0.0
        for genome in self.genomes:
            total += genome.global_rank
        self.average_fitness = total / len(self.genomes)
        return total / len(self.genomes)

    def evaluate_fitness(self):
        for genome in self.genomes:
            genome.evaluate_fitness()
        self.max_fitness = max(self.genomes, key=lambda x: x.fitness).fitness