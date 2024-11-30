import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
from simulation import generate_stations, initialize_graph, GeneticAlgorithm, station_holder, evaluate_graph
import random
import time

# Helper function to plot the graph and return the image buffer
def plot_graph(graph, stations):
    # Assign colors based on node types
    node_colors = []
    for station in stations:
        if station.type == 'Res':
            node_colors.append('red')
        elif station.type == 'Ind':
            node_colors.append('green')
        elif station.type == 'Com':
            node_colors.append('blue')
        else:
            node_colors.append('gray')  # Fallback for unknown types
    
    plt.figure(figsize=(10, 6))  # Reduced figure size for better fit
    pos = nx.get_node_attributes(graph, "pos")  # Get positions from node attributes
    
    # Validate that all nodes have positions
    if len(pos) != len(graph.nodes):
        raise ValueError("Not all nodes have positions assigned.")

    # Draw the graph with the appropriate node colors
    nx.draw(
        graph, pos, with_labels=False, node_size=50, alpha=0.7, node_color=node_colors
    )
    
    # Save the plot to a BytesIO buffer for use in Streamlit
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf


# Initialize session state
if "generation" not in st.session_state:
    st.session_state["generation"] = 0
    st.session_state["graph"] = None
    st.session_state["ga"] = None
    st.session_state["stats"] = {"best_cost": [], "generation_count": 0}

# Layout management using columns
st.set_page_config(layout="wide", page_title="Graph Evolution")

# UI layout with two columns: Graph on the left, controls on the right
col1, col2 = st.columns([3, 2])  # 3:2 ratio for left (graph) and right (controls)

# Title Section
with col1:
    st.title("Metro Optimization")

if st.session_state["generation"] == 0:
    with col2:
        st.header("Initialize Graph")
        # Sliders for Station Counts
        residential_count = st.slider("Residential Stations", 1, 50, 10)
        industrial_count = st.slider("Industrial Stations", 1, 50, 10)
        commercial_count = st.slider("Commercial Stations", 1, 50, 10)
        random_state = st.number_input("Optional Random State", value=None)
        if random_state is None:
            random_state = random.randint(1,1000)

        # Sliders for Map Settings
        map_size = 800
        cluster_spread = 50

        # Button to initialize the graph
        if st.button("Initialize Graph"):
            zones_count = {"Res": residential_count, "Ind": industrial_count, "Com": commercial_count}
            stations = generate_stations(zones_count=zones_count, random_state=random_state, map_size=map_size, cluster_spread=cluster_spread)
            graph = initialize_graph(stations)

            # Set stations in the singleton StationHolder
            station_holder.set_stations(stations)

            # Initialize Genetic Algorithm
            ga = GeneticAlgorithm(graph)
            ga.initialize_population(size=10)  # Default population size

            st.session_state["graph"] = graph
            st.session_state["ga"] = ga
            st.session_state["generation"] = 1
            st.session_state["stats"] = {"best_cost": [], "generation_count": 0}
            st.success("Graph and Genetic Algorithm Initialized!")

else:
    with col2:
        # Header for the current generation
        st.header(f"Generation {st.session_state['generation']}")

        # Display hyperparameters
        st.subheader("Adjust Hyperparameters")
        population_size = st.slider("Population Size", 5, 50, 10, key="pop_size_slider")
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.1, step=0.05, key="mutation_slider")

        # Button to evolve
        if st.button("Evolve"):
            with st.spinner("Evolving..."):
                starttime = time.time()
                # Get current graph and GA
                ga = st.session_state["ga"]

                # Evaluate current population
                ga.evaluate_population(fitness_function=lambda g: evaluate_graph(g))
                best_graph, best_cost = ga.select_best()

                # Update statistics
                st.session_state["stats"]["best_cost"].append(best_cost)
                st.session_state["stats"]["generation_count"] += 1

                # Update session state for graph and generation
                st.session_state["graph"] = best_graph
                st.session_state["generation"] += 1

                # Evolve to the next generation
                ga.evolve()
            st.success(f"Generation {st.session_state['generation'] - 1} Complete!")

        # Display statistics
        st.subheader("Stats")
        if st.session_state["stats"]["best_cost"]:
            st.metric(label="Best Cost", value=f"{st.session_state['stats']['best_cost'][-1]:.2f}")
        else:
            st.write("No statistics available yet. Click 'Evolve' to start.")

        st.metric(label="Total Generations", value=st.session_state["stats"]["generation_count"])

        try:
            st.metric(label="Compute Time", value=f"{round((time.time()-starttime), 3)}s") # label compute time if compute time exists
        except:
            pass

        # Reset Button
        if st.button("Reset"):
            st.session_state["generation"] = 0
            st.session_state["graph"] = None
            st.session_state["ga"] = None
            st.session_state["stats"] = {"best_cost": []}
            station_holder.set_stations([])  # Clear the stations
            st.success("Application Reset!")

# Display the graph on the left side
with col1:
    st.subheader("Map Display")  # Add a header for clarity
    st.text("Red: Residential\nBlue: Commercial\nGreen: Industrial")
    if st.session_state["graph"]:
        buf = plot_graph(st.session_state["graph"], station_holder.get_stations())
        st.image(buf, caption=f"Graph for Generation {st.session_state['generation']}", use_container_width=True)
