from constants import *
from genome import Genome
from species import Species
import numpy as np
from numpy.random import choice

import logging
from network import Network

class Pool(object):
    def __init__(self, population):
        logging.info("Starting Pool Initialization")

        self.innovation = 0
        self.population = population
        self.species = []
        self.generation = 0

        genome = Genome(self.new_innovation)
        genome.set_basic()

        logging.info("Adding genomes to initial pool")
        for i in range(population):
            logging.info("Adding genome %d" % i)
            self.add_to_species(genome.copy())
        logging.info("Pool initialization complete")

    def add_to_species(self, genome):
        added = False
        for specie in self.species:
            if specie.same_species(genome):
                specie.add_genome(genome)
                added = True
                break

        if not added:
            self.add_species_with_genome(genome)

    def new_innovation(self):
        self.innovation += 1
        return self.innovation

    def add_species_with_genome(self, genome):
        self.species.append(Species(genome))

    def get_all_genomes(self):
        result = []
        for specie in self.species:
            for genome in specie.genomes:
                result.append(genome)
        return result

    def rank_globally(self):
        genomes = self.get_all_genomes()
        genomes.sort(key=lambda x: x.fitness)

        for i in range(len(genomes)):
            genomes[i].global_rank = i

    def remove_stale_species(self):
        survived = []

        logging.debug("remove stale species: %d" % len(self.species))
        for specie in self.species:
            specie.genomes.sort(key=lambda x: -x.fitness)
            if specie.genomes[0].fitness > specie.top_fitness:
                specie.staleness = 0
                specie.top_fitness = specie.genomes[0].fitness
            else:
                specie.staleness += 1

            if specie.staleness < MAX_STALENESS or specie.top_fitness >= self.max_fitness:
                survived.append(specie)
        self.species = survived

    def total_average_fitness(self):
        self.rank_globally()

        total = 0.0
        for specie in self.species:
            specie.calc_average_fitness()
            total += specie.average_fitness

        return total

    def remove_weak_species(self):
        survived = []
        sum = self.total_average_fitness()

        for specie in self.species:
            if specie.average_fitness*self.population/sum >= 1.0:
                survived.append(specie)

        self.species = survived

    def evaluate_fitness(self):
        for specie in self.species:
            specie.evaluate_fitness()

        logging.debug(len(self.species))
        self.max_fitness = max(self.species, key=lambda x: x.max_fitness).max_fitness

    def evolve(self, max_gen = 100):
        for gen in range(max_gen):
            logging.info("Starting generation %d" % (gen+1))

            self.evaluate_fitness()

            logging.info("Culling half of each species")
            for specie in self.species:
                specie.cull_species(False)

            self.evaluate_fitness()
            logging.info("Removing stale species")
            self.remove_stale_species()

            self.evaluate_fitness()

            logging.info("Starting crossovers and mutations")
            children = []
            sum = self.total_average_fitness()

            for specie in self.species:
                breed = int(specie.average_fitness*self.population/sum) - 1
                for i in range(breed):
                    children.append(specie.breed_child())

            logging.info("Culling to one")
            for specie in self.species:
                specie.cull_species(True)

            logging.info("Generating more crossovers to reach population")
            while len(children) + len(self.species) < self.population:
                specie = self.species[choice(len(self.species))]
                children.append(specie.breed_child())

            for child in children:
                self.add_to_species(child)

            logging.info("Max fitness %d" % self.max_fitness)
            logging.info("Generation %d done" % (gen+1))

        genomes = self.get_all_genomes()

        logging.info("OUTPUT")
        network = Network(max(genomes, key=lambda x: x.fitness))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        logging.info("%d ^ %d ^ %d ^ %d= %d" % (i, j, k, l, network.evaluate([i,j,k,l,1])[0]))

        max(genomes, key=lambda x: x.fitness).draw()