import networkx as nx
import numpy as np
import random
from collections import deque


p_passengers = {
    ("Res", "Res"): 0.2,
    ("Ind", "Ind"): 0.1,
    ("Com", "Com"): 0.1,
    ("Res", "Ind"): 0.1,
    ("Res", "Com"): 0.1,
    ("Com", "Res"): 0.1,
    ("Ind", "Com"): 0.1,
    ("Com", "Ind"): 0.1,
    ("Ind", "Res"): 0.1
}

class TripCounter:
    def __init__(self):
        self.trips = {}
    def update(self, a, b):
        tpl = (a,b)
        if tpl not in self.trips:
            self.trips[tpl] = 1
        else:
            self.trips[tpl] += 1

class Passenger:
    current_id = 0
    known_paths = {}

    def reset_paths():
        Passenger.known_paths = {}

    def __init__(self, position_id, destination_id):
        self.passenger_id = Passenger.current_id
        self.destination_id = destination_id
        self.current_station_id = position_id
        self.ticks = 0
        self.cost = 0
        self.path = []
        self.distance_traveled = 0
        Passenger.current_id +=1
    
    def get_next_station(self, graph):
        self.path = nx.dijkstra_path(graph, self.current_station_id, self.destination_id)
            
        if len(self.path) > 1:
            return self.path[1]
        else:
            return None            

    def update_pos(self, graph, stations, trip_counter):
        if self.current_station_id == self.destination_id:
            return # implement logic to deal with finished passengers
    
        new_pos = self.get_next_station(graph)
        if new_pos is not None:
            a = self.current_station_id
            b = new_pos
            self.update_cost(graph, a, b)
            trip_counter.update(a=self.current_station_id, b=new_pos) # add trip to step's tripcounter
            self.current_station_id = new_pos

            pos_a = stations[a].position
            pos_b = stations[b].position
            self.distance_traveled += np.linalg.norm((pos_a[0]-pos_b[0], pos_a[1] - pos_b[1]))
            self.ticks+=1

        else:
            self.current_station_id = self.destination_id
    
    def update_cost(self, graph, a, b):
        weight = graph[self.current_station_id][b]['weight']
        self.cost += weight


    def arrived(self):
        return self.current_station_id == self.destination_id

    def describe(self):
        print(
            f"Passenger {self.passenger_id} at station {self.current_station_id} with destination {self.destination_id} - Elapsed {self.ticks} ticks"
        )
       
def generate_passengers(stations, p_passengers, count):
    passengers = []
    for i in range(count):
        path_type = random.choices(list(p_passengers.keys()), weights=list(p_passengers.values()))
        start, end = path_type[0][0], path_type[0][1]
        start_station = random.choice([station for station in stations if station.type == start])
        end_station = random.choice([station for station in stations if station.type == end])

        newPassenger = Passenger(start_station.station_id, end_station.station_id)

        passengers.append(newPassenger)

    return passengers

def calculate_edge_distance(a, b, stations):
    pos_a = [station for station in stations if station.station_id == a][0].position
    pos_b = [station for station in stations if station.station_id == b][0].position
    return np.linalg.norm(np.array(pos_a) - np.array(pos_b))

def orientation(p, q, r):
    """
    Calculate the orientation of the triplet (p, q, r).
    Returns:
    0 -> p, q, r are collinear
    1 -> Clockwise
    2 -> Counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2

def do_intersect(p1, q1, p2, q2):
    """
    Check if line segments (p1, q1) and (p2, q2) intersect.
    """
    # Find the four orientations needed for the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # general case
    if o1 != o2 and o3 != o4:
        return True

    # p1, q1, and p2 are collinear, and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    # p1, q1, and q2 are collinear, and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    # p2, q2, and p1 are collinear, and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    # p2, q2, and q1 are collinear, and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False

def on_segment(p, q, r):
    """
    Check if point q lies on segment pr.
    """
    if q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]):
        return True
    return False

def count_edge_overlaps(graph):
    """
    Count the number of overlapping edges in the graph.
    """
    pos = nx.get_node_attributes(graph, "pos") 
    edges = list(graph.edges)
    overlap_count = 0

    # check over all pairs of edges
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            u1, v1 = edges[i]
            u2, v2 = edges[j]

            p1, q1 = pos[u1], pos[v1]
            p2, q2 = pos[u2], pos[v2]

            if do_intersect(p1, q1, p2, q2):
                overlap_count += 1

    return overlap_count

def weighted_cost(arrived_passengers, trip_counter, graph, stations, weights):
    avg_travel_time = np.average([p.ticks for p in arrived_passengers])
    congestion_penalty = np.mean([graph.edges[a, b]['weight'] for a, b in trip_counter.trips.keys()])
    edge_use_variance = np.var([graph.edges[a, b]['weight'] for a, b in trip_counter.trips.keys()])
    edge_count = len(graph.edges)
    edge_distances = [calculate_edge_distance(a, b, stations) for a, b in graph.edges]
    overlapping_edges = count_edge_overlaps(graph)
    mean_edge_distance = np.mean(edge_distances)
    mean_traveled_distance = np.mean([p.distance_traveled for p in arrived_passengers])
    mean_distance_cost = np.mean([((10 + p.cost)/10 * p.distance_traveled) for p in arrived_passengers])    
    # combine factors and weights
    total_cost = (
        weights['AvgStops'] *(100/4.855) * avg_travel_time +            
        weights['AvgCongestion']*(100/29.86) * congestion_penalty +         
        weights['LineUseVariance']*(100/489) * edge_use_variance +
        weights['LineCount']*(100/44) * edge_count + 
        weights['AvgLineDistance']*(100/212) * mean_edge_distance + 
        weights['AvgTripDistance']*(100/2000) * mean_traveled_distance + 
        weights['MeanDistCost']*(100/800) * mean_distance_cost +
        weights['OverlappingLines']* overlapping_edges
    )  
    return total_cost

def update_edge_weights(graph, trip_counter):
    for trip in trip_counter.trips.keys():
        graph.edges[trip[0], trip[1]]['weight'] = trip_counter.trips[trip]

def batch(graph, stations, generation_quantity, iterations):
    regens = 0
    arrived_passengers = []
    passengers = deque(generate_passengers(stations, p_passengers, generation_quantity))
    trip_counter = TripCounter()
    while regens < iterations:
        
        for passenger in list(passengers):  # converted to list to safely iterate and remove

            passenger.update_pos(graph, stations, trip_counter)

            if passenger.arrived():
                arrived_passengers.append(passenger)
                passengers.remove(passenger)
        # update weights conditionally
        if regens % 5 == 0:
            update_edge_weights(graph, trip_counter)
        if regens < iterations:
            passengers.extend(generate_passengers(stations, p_passengers, generation_quantity))
        regens += 1
    return arrived_passengers, trip_counter, graph

def evaluate_state(graph, stations, weights, evaluation_function=weighted_cost):
    arrivals, trip_counter, graph = batch(graph, stations, generation_quantity=20, iterations=40)
    cost = evaluation_function(arrivals, trip_counter, graph, stations, weights)
    return cost

def secondary_evaluation(graph, stations):
    arrived_passengers, trip_counter, graph = batch(graph, stations, generation_quantity=20, iterations=40)

    avg_travel_time = np.average([p.ticks for p in arrived_passengers])
    congestion_penalty = np.mean([graph.edges[a, b]['weight'] for a, b in trip_counter.trips.keys()])
    edge_use_variance = np.var([graph.edges[a, b]['weight'] for a, b in trip_counter.trips.keys()])
    edge_count = len(graph.edges)
    overlapping_edges = count_edge_overlaps(graph)
    mean_edge_distance = np.mean([calculate_edge_distance(a, b, stations) for a, b in graph.edges])
    mean_traveled_distance = np.mean([p.distance_traveled for p in arrived_passengers])

    return {
        "AvgTravelStops": avg_travel_time,
        "AvgCongestion": congestion_penalty,
        "LineUseVariance": edge_use_variance,
        "LineCount": edge_count,
        "NumOverlappingLines": overlapping_edges,
        "AvgLineDist": mean_edge_distance,
        "AvgTripDist": mean_traveled_distance
    }