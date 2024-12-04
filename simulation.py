import networkx as nx
import numpy as np
import random
import batch
from scipy.spatial import KDTree


class Station:
    def __init__(self, id, position):
        self.station_id = id
        self.position = position
        self.distances = None

    def set_distances(self, distances):
        self.distances = distances

class StationHolder:
    def __init__(self):
        self.stations = []

    def set_stations(self, stations):
        self.stations = stations

    def get_stations(self):
        return self.stations

station_holder = StationHolder()

def calculate_all_station_distances(stations):
    positions = np.array([station.position for station in stations])
    station_ids = [station.station_id for station in stations]

    # compute pairwise distances
    distances_matrix = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)

    # set diagonal to inf
    np.fill_diagonal(distances_matrix, np.inf)

    # convert distances to dict
    distances_dict = {
        station_id: {
            other_id: distances_matrix[i, j] 
            for j, other_id in enumerate(station_ids)
        }
        for i, station_id in enumerate(station_ids)
    }

    return distances_dict

def generate_stations(station_count, random_state, map_size=800, cluster_spread=50, min_distance=20):
    rng = np.random.default_rng(random_state)
    stations = []
    center = rng.random(2) * map_size 
    positions = []  # store station positions for KDTree

    while len(stations) < station_count:
        location = center + rng.normal(0, cluster_spread, 2)

        # use KDTree to check minimum distance constraint
        if positions:
            tree = KDTree(positions)
            if tree.query_ball_point(location, min_distance):
                continue

        # add the station if no neighbors violate the constraint
        station = Station(len(stations), location)
        stations.append(station)
        positions.append(location)  # update KDTree positions

    
    all_distances = calculate_all_station_distances(stations)
    for station in stations:
        if station.station_id in all_distances:
            station.set_distances(all_distances[station.station_id])
        else:
            raise ValueError
        
    return stations


def initialize_graph(stations):
    graph = nx.Graph()  
    for station in stations:
        graph.add_node(station.station_id, pos=station.position)
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
        
        for node in nodes:
            offspring.add_node(node, **parent1.nodes[node])

        parent_edges = list(parent1.edges) + list(parent2.edges)
        total_edges = min(self.max_edges, len(parent_edges))

        if total_edges > 0:
            selected_edges = random.sample(parent_edges, total_edges)  # No need for numpy here
            offspring.add_edges_from((u, v, {"weight": 0}) for u, v in selected_edges)

        if not nx.is_connected(offspring):
            components = list(nx.connected_components(offspring))
            for i in range(len(components) - 1):
                node1 = random.choice(list(components[i]))
                node2 = random.choice(list(components[i + 1]))
                offspring.add_edge(node1, node2, weight=0)

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


    def evolve(self, mutation_rate, children_count):
        """Evolves the population to produce the next generation."""
        next_generation = []
        for _ in range(children_count):
            parent1, parent2 = random.sample(self.offspring, 2)
            child = self.generate_offspring(parent1, parent2, mutation_rate)
            next_generation.append(child)
        self.offspring = next_generation


    def auto_evolve(self):
        """Automatically evolves graph and stops when improvement is minimal."""
        mutation_rate = 0.4
        children_count = 20
        gen = 1
        patience = 10  
        generations_since_improvement = 0
        weights_dict = {
            'MeanDistCost': 1.0,
            'AvgStops': 0.0,
            'AvgCongestion': 0.0,
            'LineUseVariance': 0.0,
            'LineCount': 0.0,
            'AvgLineDistance': 0.0,
            'AvgTripDistance': 0.0,
            'OverlappingLines': 0.0
        }
        history = {}
        best_cost_overall = float("inf")  
        best_graph_overall = None 

        while True:
            mutation_rate = max(0.01, mutation_rate * 0.96)
            
            self.evolve(mutation_rate, children_count)
            self.evaluate_population(fitness_function=lambda g: evaluate_graph(g, weights_dict))
            best_graph, best_cost = self.select_best()
            
            if best_cost < best_cost_overall:
                best_cost_overall = best_cost
                best_graph_overall = best_graph
                generations_since_improvement = 0 
            else:
                generations_since_improvement += 1

            history[gen] = (best_graph, best_cost)

            if generations_since_improvement >= patience:
                return best_graph_overall, best_cost_overall, history

            gen += 1


