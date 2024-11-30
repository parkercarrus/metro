import networkx as nx
import numpy as np
import random
import batch

P_CAPACITY = {"Res": (100, 10), "Com": (200, 20), "Ind": (150, 15)}

class Station:
    def __init__(self, id, position, type, p_capacity=P_CAPACITY):
        self.station_id = id
        self.position = position
        self.type = type
        self.capacity = int(np.random.normal(p_capacity[self.type][0], p_capacity[self.type][1]))

class StationHolder:
    def __init__(self):
        self.stations = []

    def set_stations(self, stations):
        self.stations = stations

    def get_stations(self):
        return self.stations

station_holder = StationHolder()


def generate_stations(zones_count, random_state, map_size=800, cluster_spread=50):
    rng = np.random.default_rng(random_state) 

    stations = []
    for station_type, num_clusters in zones_count.items():
        for _ in range(num_clusters):
            center = rng.random(2) * map_size  # generate random cluster center
            location = center + rng.normal(0, cluster_spread, 2)  # cluster spread
            station = Station(len(stations), location, station_type)
            stations.append(station)
    return stations


def initialize_graph(stations):
    graph = nx.Graph()
    for station in stations:
        graph.add_node(station.station_id, pos=station.position, type=station.type)
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            # add edge
            graph.add_edge(stations[i].station_id, stations[j].station_id, weight=random.uniform(0, 1))
            break
    return graph

def evaluate_graph(graph, weights):
    stations = station_holder.get_stations()
    return batch.evaluate_state(graph, stations, weights)

class GeneticAlgorithm:
    def __init__(self, graph, max_edges):
        self.graph = graph
        self.max_edges = max_edges
        self.offspring = []
        self.fitness = None

    def initialize_population(self, size, mutation_rate=0.8):
        """Generates the initial population of offspring."""
        for _ in range(size):
            child_graph = self.generate_offspring(self.graph, self.graph, mutation_rate)
            self.offspring.append(child_graph)

    def generate_offspring(self, parent1, parent2, mutation_rate):
        nodes = parent1.nodes
        offspring = nx.Graph()
        # add nodes and preserve their positions
        for node in nodes:
            offspring.add_node(node, **parent1.nodes[node]) 

        # combine edges from both parents, limited to the edge budget
        total_edges = min(len(parent1.edges) + len(parent2.edges), self.max_edges)
        edges = random.sample(list(parent1.edges) + list(parent2.edges), total_edges)
        offspring.add_edges_from((u, v, {"weight": 0}) for u, v in edges)

        # ensure offspring is connected
        if not nx.is_connected(offspring):
            components = list(nx.connected_components(offspring))
            for i in range(len(components) - 1):
                node1 = random.choice(list(components[i]))
                node2 = random.choice(list(components[i + 1]))
                offspring.add_edge(node1, node2, weight=0)

        # apply mutation
        self.mutate_offspring(offspring, mutation_rate)
        return offspring


    def mutate_offspring(self, offspring, mutation_rate):
        """
        Mutates the offspring by adding or removing edges.
        """
        if random.random() < mutation_rate:
            if offspring.number_of_edges() < self.max_edges:
                # add a new edge if under limit
                u, v = random.sample(list(offspring.nodes()), 2) 
                if not offspring.has_edge(u, v):
                    offspring.add_edge(u, v, weight=0)
            elif offspring.number_of_edges() > self.max_edges:
                # remove a random edge if over limit
                u, v = random.choice(list(offspring.edges))
                offspring.remove_edge(u, v)


    def evaluate_population(self, fitness_function=evaluate_graph):
        """Evaluates the fitness of each offspring in the population."""
        self.fitness = [fitness_function(graph) for graph in self.offspring]

    def select_best(self):
        """Selects the best offspring based on fitness."""
        best_idx = min(range(len(self.fitness)), key=self.fitness.__getitem__)
        return self.offspring[best_idx], self.fitness[best_idx]

    def evolve(self, mutation_rate, children_count, weights):
        """Evolves the population to produce the next generation."""
        next_generation = []
        for _ in range(children_count):
            parent1, parent2 = random.sample(self.offspring, 2)
            child = self.generate_offspring(parent1, parent2, mutation_rate)
            next_generation.append(child)
        self.offspring = next_generation