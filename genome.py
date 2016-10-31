from constants import *
from numpy.random import random
from numpy.random import uniform
from numpy.random import choice

from network import Network
import networkx as nx
import matplotlib.pyplot as plt

def pcount(x):
    result = 0
    while x>0:
        if random() < x:
            result += 1
        x -= 1
    return result


class Genome(object):
    def __init__(self, global_innovation_func):
        self.genes = []
        self.fitness = 0.0
        self.max_neuron = INPUT_SIZE
        self.get_global_innovation = global_innovation_func
        self.mutation_rates = {}
        self.mutation_rates['connect'] = MUTATE_CONNECTIONS_CHANCE
        self.mutation_rates['link'] = LINK_MUTATION_CHANCE
        self.mutation_rates['bias'] = BIAS_MUTATION_CHANCE
        self.mutation_rates['node'] = NODE_MUTATION_CHANCE
        self.mutation_rates['enable'] = ENABLE_MUTATION_CHANCE
        self.mutation_rates['disable'] = DISABLE_MUTATION_CHANCE
        self.mutation_rates['step'] = STEP_MUTATION

    def copy(self):
        result = Genome(self.get_global_innovation)
        result.fitness = self.fitness
        result.max_neuron = self.max_neuron
        result.mutation_rates = {}
        result.genes = []
        for gene in self.genes:
            result.genes.append(gene.copy())
        for mutation, rate in self.mutation_rates.iteritems():
            result.mutation_rates[mutation] = rate
        return result

    def set_basic(self):
        for i in range(1, INPUT_SIZE+1):
            for j in range(1, OUTPUT_SIZE+1):
                self.genes.append(Gene(enabled=True, weight=uniform(-2,2), in_node=i,
                                       out_node=MAX_NODES+j, innovation=self.get_global_innovation()))
        return self

    def crossover(self, a):
        if self.fitness < a:
            g1 = a
            g2 = self
        else:
            g1 = self
            g2 = a

        child = Genome(self.get_global_innovation)
        innovation_map = {}

        for gene in g2.genes:
            innovation_map[gene.innovation] = gene

        for gene in g1.genes:
            if gene.innovation in innovation_map:
                if random > 0.5:
                    child.genes.append(gene.copy())
                else:
                    child.genes.append(innovation_map[gene.innovation].copy())
            else:
                child.genes.append(gene.copy())

        child.max_neuron = max(g1.max_neuron, g2.max_neuron)
        child.mutation_rates = {}
        for mutation, rate in g1.mutation_rates.iteritems():
            child.mutation_rates[mutation] = rate

        return child

    def point_mutate(self):
        step = self.mutation_rates['step']

        for gene in self.genes:
            if random() < PERTURB_CHANCE:
                gene.weight = gene.weight + uniform(-step, step)
            else:
                gene.weight = uniform(-2, 2)

    def enable_disable_mutate(self, enable):
        candidates = []
        for gene in self.genes:
            if gene.enabled != enable:
                candidates.append(gene)

        if candidates:
            candidates[choice(len(candidates))].enabled = enable

    def node_mutate(self):
        if not self.genes:
            return

        gene = self.genes[choice(len(self.genes))]
        if not gene.enabled:
            return

        gene.enabled = False
        self.max_neuron += 1

        gene1 = Gene(enabled=True, weight=1.0, in_node=gene.into,
                     out_node=self.max_neuron, innovation=self.get_global_innovation())
        gene2 = Gene(enabled=True, weight=gene.weight, in_node=self.max_neuron,
                     out_node=gene.out, innovation=self.get_global_innovation())

        self.genes.append(gene1)
        self.genes.append(gene2)

    def contains_link(self, link):
        for gene in self.genes:
            if gene.into == link.into and gene.out == link.out:
                return True
        return False

    def get_random_neuron(self, include_input_neurons):
        candidates = {}

        if include_input_neurons:
            for i in range(1, INPUT_SIZE):
                candidates[i] = True

        for i in range(1, OUTPUT_SIZE+1):
            candidates[MAX_NODES + i] = True

        for gene in self.genes:
            if include_input_neurons or gene.out > INPUT_SIZE:
                candidates[gene.out] = True

            if include_input_neurons or gene.into > INPUT_SIZE:
                candidates[gene.into] = True

        random_index = choice(len(candidates))
        return candidates.keys()[random_index]

    def link_mutate(self, force_bias):
        neuron1 = self.get_random_neuron(include_input_neurons=False)
        neuron2 = self.get_random_neuron(include_input_neurons=True)

        if (neuron1 <= INPUT_SIZE and neuron2 <= INPUT_SIZE) or neuron1 == neuron2:
            return

        if neuron2 <= INPUT_SIZE or neuron1 > MAX_NODES:
            temp = neuron1
            neuron1 = neuron2
            neuron2 = temp



        if force_bias:
            neuron1 = INPUT_SIZE

        new_gene = Gene(enabled=True, weight=uniform(-2.0,2.0), in_node=neuron1, out_node=neuron2)

        if self.contains_link(new_gene):
            return
        else:
            new_gene.innovation = self.get_global_innovation()

        self.genes.append(new_gene)

    def mutate(self):
        for mutation, rate in self.mutation_rates.iteritems():
            if random() < 0.5:
                self.mutation_rates[mutation] *= 0.95
            else:
                self.mutation_rates[mutation] *= 1.05263

        for i in range(pcount(self.mutation_rates['connect'])):
            self.point_mutate()

        for i in range(pcount(self.mutation_rates['link'])):
            self.link_mutate(force_bias=False)

        for i in range(pcount(self.mutation_rates['bias'])):
            self.link_mutate(force_bias=True)

        for i in range(pcount(self.mutation_rates['node'])):
            self.node_mutate()

        for i in range(pcount(self.mutation_rates['enable'])):
            self.enable_disable_mutate(enable=True)

        for i in range(pcount(self.mutation_rates['disable'])):
            self.enable_disable_mutate(enable=False)

    def disjoint(self, genome_b):
        innovation_map = {}
        for gene in self.genes:
            innovation_map[gene.innovation] = True

        total_common = 0
        for gene in genome_b.genes:
            if gene.innovation in innovation_map:
                total_common += 1

        total_disjoint = len(self.genes) + len(genome_b.genes) - 2 * total_common
        n = max(len(self.genes), len(genome_b.genes))

        return total_disjoint*1.0/n

    def diff_weights(self, genome_b):
        innovation_map = {}
        for gene in self.genes:
            innovation_map[gene.innovation] = gene.weight

        total_common = 0
        sum = 0.0
        for gene in genome_b.genes:
            if gene.innovation in innovation_map:
                sum += abs(gene.weight - innovation_map[gene.innovation])
                total_common += 1

        return sum/total_common

    def same_species(self, genome_b):
        dd = DELTA_DISJOINT * self.disjoint(genome_b=genome_b)
        dw = DELTA_WEIGHT * self.disjoint(genome_b=genome_b)
        return dd + dw < DELTA_THRESHOLD

    def evaluate_fitness(self):
        network = Network(self)

        total = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        output = network.evaluate([i, j, k, l, 1])
                        total += abs(output[0] - (i ^ j ^ k ^ l))

        result = (16-total)**2
        self.fitness = result
        return result

    def draw(self):
        G = nx.Graph()

        nodes_added = {}
        labels = {}

        io_pos = {}
        for i in range(1, INPUT_SIZE+1):
            io_pos[i] = (-20,i)

        for i in range(1, OUTPUT_SIZE+1):
            io_pos[MAX_NODES + i] = (10, i)

        for gene in self.genes:
            into = gene.into
            out = gene.out
            if into not in nodes_added:
                nodes_added[into] = True
                G.add_node(into)
                labels[into] = into
            if out not in nodes_added:
                nodes_added[out] = True
                G.add_node(out)
                labels[out] = out

            if gene.enabled:
                G.add_edge(into, out, color='green')
            else:
                G.add_edge(into, out, color='red')

        pos = nx.spring_layout(G, pos=io_pos, fixed=io_pos.keys())

        colors = [G[u][v]['color'] for u, v in G.edges()]
        nx.draw_networkx_nodes(G, pos, with_labels = False, node_color='b')
        nx.draw_networkx_edges(G, pos, edge_color=colors)
        nx.draw_networkx_labels(G, pos, labels, font_color='w')
        plt.show(block = True)

class Gene(object):
    def __init__(self, enabled=False, weight=0.0, in_node=0, out_node=0, innovation = 0):
        self.enabled = enabled
        self.weight = weight
        self.into = in_node
        self.out = out_node
        self.innovation = innovation

    def copy(self):
        return Gene(self.enabled, self.weight, self.into, self.out, self.innovation)

