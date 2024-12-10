import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from nsga_II import nsga_ii, plot_fitness_history

# Title of the app
st.title("Specifying the problem")

# Input array from the user
objective1_coefficient = st.text_input("Enter an coefficient of variables objective 1(comma-separated values, 0 if it dosent exist):")
objective2_coefficient = st.text_input("Enter an coefficient of variables objective 2(comma-separated values, 0 if it dosent exist):")
constraint_coefficient = st.text_input("Enter an coefficient of variables constraints(comma-separated values):")
col1, col2, col3, col4 = st.columns(4)
with col1:
    borne_constraint = st.number_input("Constraints borne:", value=0)
with col2:
    num_items = st.number_input("Variables:", value=0)
with col3:
    num_generations = st.number_input("Generations:", value=0)
with col4:
    population_size = st.number_input("Population size:", value=0)

# Convert the input string to a list
if objective1_coefficient and objective2_coefficient and constraint_coefficient and borne_constraint and num_items and num_generations and population_size:
    objective1_array = [int(x) for x in objective1_coefficient.split(',')]
    objective2_array = [int(x) for x in objective2_coefficient.split(',')]
    constraint_array = [int(x) for x in constraint_coefficient.split(',')]

    items = []
    for i in range(num_items):
        item = {
            "name": f"item{i+1}",
            "weight": constraint_array[i] if i < len(constraint_array) else 0,
            "value": [
                objective1_array[i] if i < len(objective1_array) else 0,
                objective2_array[i] if i < len(objective2_array) else 0
            ]
        }
        items.append(item)

    # Call the NSGA-II algorithm
    fitness_history, solution_history = nsga_ii(population_size, items, borne_constraint, 2, num_generations, 0.5, 0.05, num_items)	
    
    plot_fitness_history(fitness_history)
    sums = [np.sum(pair) for pair in fitness_history]
    best_index = np.argmax(sums)
    print(fitness_history[best_index])
    print(solution_history[best_index])
    st.write("Best solution found:")
    st.text(solution_history[best_index])
    st.write("Fitness:")
    st.text(fitness_history[best_index])


# n_items = 10
# weights = np.array([3, 4, 2, 7, 5, 6, 1, 8, 9, 4])
# values = np.array([15, 25, 10, 30, 20, 35, 5, 40, 45, 10])
# costs = np.array([-8, -10, -6, -12, -9, -15, -7, -11, -14, -8])
# capacity = 30