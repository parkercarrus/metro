import streamlit as st
import module

st.set_page_config(layout="wide", page_title="Metro Optimization Evolution")
tabs = st.tabs(["Use", "About"])

with tabs[0]:
    module.main()

with tabs[1]:
    st.title("About this App")
    st.markdown("""
    ## 🚇 An Algorithmic Approach to Metro Planning

    This application employs **genetic algorithms** to design and optimize public transit networks based on a randomly generated set of stations. The system aims to create efficient, balanced metro maps by considering factors such as:
    - 🚦 **Line Congestion**
    - ⏱️ **Passenger Travel Time**
    - 📏 **Line Distance**
    - ...and more.

    ---

    ## 🔑 Key Features
    - **Customizable Parameters**: Adjust mutation rates, population size, and evaluation metrics to tailor the optimization process.
    - **Interactive Visualization**: View the evolving metro map in real time with intuitive color-coded nodes and edges.
    - **Batch Evolution**: Progress through multiple generations in one go and monitor key metrics across iterations.

    ---

    ## 🧬 Genetic Algorithm Overview
    The optimization process is driven by a genetic algorithm, which simulates evolution by:
    1. **Evaluating Fitness**: Using a user-defined objective function to score each metro map.
    2. **Selection**: Choosing the best-performing maps to propagate.
    3. **Mutation & Crossover**: Introducing variations to generate diverse and improved solutions.

    ---

    ## 📊 Evaluation of a Metro Map
    Each metro map (individual) is assessed by simulating passenger transport over a set **epoch**. The evaluation process includes:
    - **Passenger Grouping**: Passengers are categorized by their origin and destination stations.
    - **Passenger Destinations** Passengers are more likely to choose closer stations
    - **Passenger Generation**: Each station generates passengers probabilistically, following a normal distribution influenced by the station's distance from the map center.
    - **Performance Metrics**: Metrics like average travel time, congestion, and line usage are calculated to determine overall map efficiency.

    ## 🔜 Next Steps
    - **Continuous Lines**: Currently, lines exist independently between stations and the system does not account for transfer times.
    - **Real-World Data Integration**: Working to download real-world metro maps and evaluate this algorithm upon them.
    ---

    ## 👤 Author
    Developed by [Parker Carrus](https://linkedin.com/in/parkercarrus).

    """)