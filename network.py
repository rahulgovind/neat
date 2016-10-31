from constants import *
import numpy as np

def sigmoid(x):
    return 2.0/(1 + np.exp(-4.9*x)) - 1.0


class Network(object):
    def __init__(self, genome):
        self.network = {}
        self.num_inputs = INPUT_SIZE
        self.num_outputs = OUTPUT_SIZE

        # Generate input neuron
        for i in range(1, INPUT_SIZE+1):
            self.network[i] = Neuron()

        # Generate output neurons
        for i in range(1, OUTPUT_SIZE+1):
            self.network[MAX_NODES + i] = Neuron()

        # Generate connections from genome
        genome.genes.sort(key=lambda x: x.out)

        for gene in genome.genes:
            if gene.enabled:
                if gene.into not in self.network:
                    self.network[gene.into] = Neuron()
                if gene.out not in self.network:
                    self.network[gene.out] = Neuron()

                self.network[gene.out].add_incoming(gene)

    def evaluate(self, inputs):
        assert(len(inputs) == self.num_inputs)

        for i in range(1, self.num_inputs):
            self.network[i].value = inputs[i]

        for key, neuron in self.network.iteritems():
            sum = 0.0

            for incoming in neuron.incoming:
                sum += incoming.weight * self.network[incoming.into].value

            if neuron.incoming:
                neuron.value = sigmoid(sum)

        result = []
        for i in range(1, OUTPUT_SIZE+1):
            result.append(self.network[MAX_NODES + i].value > 0.0)

        return result


class Neuron(object):
    def __init__(self):
        self.incoming = []
        self.value = 0.0

    def add_incoming(self, gene):
        self.incoming.append(gene)
