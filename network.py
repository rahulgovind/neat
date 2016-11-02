from constants import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1 + np.exp(-5.0*x))


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
            if gene.into not in self.network:
                self.network[gene.into] = Neuron()
            if gene.out not in self.network:
                self.network[gene.out] = Neuron()

            self.network[gene.out].add_incoming(gene)

    def evaluate(self, inputs):
        for key, neuron in self.network.iteritems():
            neuron.value = 0.0

        assert(len(inputs) == self.num_inputs)

        for i in range(self.num_inputs):
            self.network[i+1].value = inputs[i]*1.0

        for key, neuron in self.network.iteritems():
            sum = 0.0
            flag = False

            for incoming in neuron.incoming:
                if incoming.enabled:
                    sum += incoming.weight * self.network[incoming.into].value
                    flag = True

            if flag:
                neuron.value = sigmoid(sum)

        result = []
        for i in range(1, OUTPUT_SIZE+1):
            result.append(self.network[MAX_NODES + i].value)

        return result

    def draw(self):
        G = nx.DiGraph()

        nodes_added = {}
        labels = {}
        edge_labels = {}

        for key, neuron in self.network.iteritems():
            for incoming in neuron.incoming:
                into = incoming.into
                out = incoming.out
                if into not in nodes_added:
                    nodes_added[into] = True
                    G.add_node(into)
                    labels[into] = into
                if out not in nodes_added:
                    nodes_added[out] = True
                    G.add_node(out)
                    labels[out] = out

                if incoming.enabled:
                    G.add_edge(into, out, color='green')
                else:
                    G.add_edge(into, out, color='red')
                edge_labels[(into, out)] = "%.3f" % incoming.weight

        pos = nx.spring_layout(G)

        colors = [G[u][v]['color'] for u, v in G.edges()]
        nx.draw_networkx_nodes(G, pos, node_color='b')
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=colors, arrows=True)
        nx.draw_networkx_labels(G, pos, labels, font_color='w')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=16)
        plt.show(block=True)



class Neuron(object):
    def __init__(self):
        self.incoming = []
        self.value = 0.0

    def add_incoming(self, gene):
        self.incoming.append(gene)
