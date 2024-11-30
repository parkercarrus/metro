import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
from simulation import generate_stations, initialize_graph, GeneticAlgorithm, station_holder, evaluate_graph
import random
import time
import batch
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


def main():

    def plot_graph(graph, stations):
        node_colors = []
        for station in stations:
            if station.type == 'Res':
                node_colors.append('red')
            elif station.type == 'Ind':
                node_colors.append('green')
            elif station.type == 'Com':
                node_colors.append('blue')
        
        plt.figure(figsize=(10, 6)) 
        pos = nx.get_node_attributes(graph, "pos")  # get positions

        if len(pos) != len(graph.nodes):
            raise ValueError("Not all nodes have positions assigned.")
        
        edge_weights = nx.get_edge_attributes(graph, 'weight') 
        if edge_weights:
            weights = list(edge_weights.values())
            min_weight, max_weight = min(weights), max(weights)
            norm = plt.Normalize(vmin=min_weight, vmax=max_weight)
            edge_colors = [
                mcolors.to_hex(plt.cm.RdYlGn(1 - norm(weight)))  # ryg colormap
                for weight in weights
            ]
        else:
            edge_colors = "gray" 

        nx.draw(
            graph, pos, with_labels=False, node_size=50, alpha=0.7, node_color=node_colors, edge_color=edge_colors
        )

        legend_elements = [
            mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10, label='Residential Station'),
            mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=10, label='Industrial Station'),
            mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=10, label='Commercial Station'),
            mpatches.Patch(color='green', label='Not Congested'),
            mpatches.Patch(color='red', label='Congested'),
        ]
        plt.legend(handles=legend_elements, loc='upper left', title="Legend")

        # save plot for streamlit use
        buf = BytesIO()
        plt.savefig(buf, format="png", facecolor='#d1d1d1')
        plt.close()
        buf.seek(0)
        return buf

    if "results_df" not in st.session_state:
        st.session_state["results_df"] = pd.DataFrame(columns=[
            "Gen",
            "Score",
            "Time",
            "AvgStops",
            "AvgCongestion",
            "LineUseVariance",
            "LineCount",
            "OverlappingLines",
            "AvgLineDistance",
            "AvgTripDistance"
        ])

    # init session state
    if "generation" not in st.session_state:
        st.session_state["generation"] = 0
        st.session_state["graph"] = None
        st.session_state["ga"] = None
        st.session_state["stats"] = {"best_cost": [], "generation_count": 0}

    # layout with two columns: Graph on the left, controls on the right
    col1, col2 = st.columns([3, 2])  

    with col1:
        st.title("Map")

    if st.session_state["generation"] == 0:
        with col2:
            st.header("Initialize Graph")
            residential_count = st.slider("Residential Stations", 1, 50, 10)
            industrial_count = st.slider("Industrial Stations", 1, 50, 10)
            commercial_count = st.slider("Commercial Stations", 1, 50, 10)
            total_stations = residential_count+industrial_count+commercial_count
            min_lines = total_stations-1
            maximum_lines = st.slider("Maximum Lines", int(min_lines), round(5*min_lines), 2*total_stations)
            random_state = st.number_input("Optional Random State", value=None)
            if random_state is None:
                random_state = random.randint(1,1000)
            else:
                random_state = int(random_state)

            map_size = 800
            cluster_spread = 50

            if st.button("Initialize Graph"):
                zones_count = {"Res": residential_count, "Ind": industrial_count, "Com": commercial_count}
                stations = generate_stations(zones_count=zones_count, random_state=random_state, map_size=map_size, cluster_spread=cluster_spread)
                graph = initialize_graph(stations)

                station_holder.set_stations(stations)

                # init Genetic Algorithm
                ga = GeneticAlgorithm(graph, maximum_lines)
                ga.initialize_population(size=10)  

                st.session_state["graph"] = graph
                st.session_state["ga"] = ga
                st.session_state["generation"] = 1
                st.session_state["stats"] = {"best_cost": [], "generation_count": 0}
                st.success("Graph and Genetic Algorithm Initialized!")

    else:
        with col2:
            st.header(f"Generation {st.session_state['generation']}")

            # display hyperparameters
            population_size = st.slider("Children Count", 5, 100, 10, key="pop_size_slider")
            mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.1, step=0.05, key="mutation_slider")
            generations_to_evolve = st.slider("Generations to Evolve", 1, 20, 1, key="generations_slider")  # batch generations

            st.caption("Evaluation Parameter Weights")
            weights_dict = {}
            with st.expander("Adjust Weights", expanded=False):
                weights_dict['MeanDistCost'] = st.slider("SmartCost Weight", 0.0, 1.0, 1.0, step=0.1)
                weights_dict['AvgStops'] = st.slider("AvgStops", 0.0, 1.0, 0.1, step=0.1)
                weights_dict['AvgCongestion'] = st.slider("AvgCongestion", 0.0, 1.0, 0.1, step=0.1)
                weights_dict['LineUseVariance'] = st.slider("LineUseVariance Weight", 0.0, 1.0, 0.1, step=0.1)
                weights_dict['LineCount'] = st.slider("LineCount Weight", 0.0, 1.0, 0.1, step=0.1)
                weights_dict['AvgLineDistance'] = st.slider("AvgLineDistance Weight", 0.0, 1.0, 0.1, step=0.1)
                weights_dict['AvgTripDistance'] = st.slider("AvgTripDistance Weight", 0.0, 1.0, 0.1, step=0.1)
                weights_dict['OverlappingLines'] = st.slider("OverlappingLines Weight", 0.0, 1.0, 0.1, step=0.1)

            # evolve button
            if st.button("Evolve"):
                with st.spinner("Evolving..."):
                    starttime = time.time()
                    ga = st.session_state["ga"]

                    for _ in range(generations_to_evolve):
                        # evaluate and evolve the population
                        ga.evaluate_population(fitness_function=lambda g: evaluate_graph(g, weights_dict))
                        best_graph, best_cost = ga.select_best()

                        # update statistics
                        st.session_state["stats"]["best_cost"].append(best_cost)
                        st.session_state["stats"]["generation_count"] += 1

                        st.session_state["graph"] = best_graph
                        st.session_state["generation"] += 1

                        # evolve to next generation
                        ga.evolve(mutation_rate, population_size, weights_dict)

                    st.success(f"Evolved through {generations_to_evolve} generations!")

            with col2:
                try:
                    st.subheader("Evaluation Results")
                    # collect results from secondary evaluation
                    secondary_evaluation_results = batch.secondary_evaluation(best_graph, station_holder.get_stations())

                    generation_number = st.session_state["generation"] - 1 
                    current_score = st.session_state["stats"]["best_cost"][-1]
                    compute_time = round((time.time() - starttime), 3)

                    current_row = {
                        "Gen": generation_number,
                        "Score": current_score,
                        "Time": compute_time,
                        "AvgStops": secondary_evaluation_results["AvgTravelStops"],
                        "AvgCongestion": secondary_evaluation_results["AvgCongestion"],
                        "LineUseVariance": secondary_evaluation_results["LineUseVariance"],
                        "LineCount": secondary_evaluation_results["LineCount"],
                        "OverlappingLines": secondary_evaluation_results["NumOverlappingLines"],
                        "AvgLineDistance": secondary_evaluation_results["AvgLineDist"],
                        "AvgTripDistance": secondary_evaluation_results["AvgTripDist"]
                    }

                    # append new stats to session df
                    st.session_state["results_df"] = pd.concat([
                        st.session_state["results_df"],
                        pd.DataFrame([current_row])
                    ], ignore_index=True)

                    st.dataframe(
                        st.session_state["results_df"],
                        use_container_width=True,
                        hide_index=True
        
                    )
                except:
                    pass


            # reset Button
            if st.button("Reset"):
                st.session_state["generation"] = 0
                st.session_state["graph"] = None
                st.session_state["ga"] = None
                st.session_state["stats"] = {"best_cost": []}
                st.session_state["results_df"] = None
                station_holder.set_stations([]) # clear stations
                st.success("Application Reset!")

    # display graph
    with col1:
        if st.session_state["graph"]:
            buf = plot_graph(st.session_state["graph"], station_holder.get_stations())
            st.image(buf, caption=f"Graph for Generation {st.session_state['generation']}", use_container_width=True)
