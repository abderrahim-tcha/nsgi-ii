import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def initialize_population(pop_size, num_items):
    population = np.array([np.random.randint(2, size=num_items) for _ in range(pop_size)])
    return population

def fitness_function(population, items, max_capacity, num_objectives, penalty_factor=10):
    fitness = np.zeros((population.shape[0], num_objectives))
    
    # For each solution in the population, calculate the total weight and value
    for i in range(population.shape[0]):
        weight = 0
        value = np.zeros(num_objectives)
        
        for j in range(population.shape[1]):
            if population[i][j] == 1:
                weight += items[j]["weight"]
                value += items[j]["value"]
        
        if weight > max_capacity:
            # Apply a penalty proportional to the excess weight
            excess_weight = weight - max_capacity
            penalty = penalty_factor * excess_weight
            fitness[i] = value - penalty
        else:
            # Assign the total value for feasible solutions
            fitness[i] = value
    return fitness

def normalize_fitness(fitness):
    min_vals = fitness.min(axis=0)
    max_vals = fitness.max(axis=0)
    normalized_fitness = (fitness - min_vals) / (max_vals - min_vals)
    return normalized_fitness

def tournament_selection(population, fitness, tournament_size):
    #summ fitness for each solution to simplyfy the selection process in multiple objectives
    summed_fitness = fitness.sum(axis=1)
    selected_parents = np.zeros((population.shape[0], population.shape[1]), dtype=int)
    for i in range(population.shape[0]):
        #chose random solutions from the population to compete in the tournament
        tournament = np.random.choice(population.shape[0], tournament_size, replace=False)
        selected_parents[i] = population[tournament[np.argmax(summed_fitness[tournament])]]
    return selected_parents

def crossover(parents, crossover_rate=0.7):
    children = np.zeros(parents.shape, dtype=int)
    for i in range(0, parents.shape[0], 2):
        if np.random.rand() < crossover_rate:
            if parents.shape[1] > 2:
                crossover_point = np.random.randint(1, parents.shape[1] - 1)
            else:
                crossover_point = 1
            children[i] = np.concatenate((parents[i][:crossover_point], parents[i + 1][crossover_point:]))
            children[i + 1] = np.concatenate((parents[i + 1][:crossover_point], parents[i][crossover_point:]))
        else:
            children[i] = parents[i]
            children[i + 1] = parents[i + 1]
    return children

def mutate(children, mutation_rate=0.01):
    for i in range(children.shape[0]):
        for j in range(children.shape[1]):
            if np.random.rand() < mutation_rate:
                children[i][j] = 1 if children[i][j] == 0 else 0
    return children

def plot_objectives(population, items, max_capacity, pareto_fronts):
    fitness = fitness_function(population, items, max_capacity, 2)
    colors = plt.cm.jet(np.linspace(0, 1, len(pareto_fronts)))
    
    for i, front in enumerate(pareto_fronts):
        front_fitness = fitness[front]
        plt.scatter(front_fitness[:, 0], front_fitness[:, 1], color=colors[i], label=f'Front {i+1}')
        # Sort the front points for better line plotting
        sorted_front = front_fitness[np.argsort(front_fitness[:, 0])]
        plt.plot(sorted_front[:, 0], sorted_front[:, 1], color=colors[i])
    
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("Objective Space")
    plt.legend()
    plt.show()

def plot_fitness_history(fitness_history):
    fitness_history = np.array(fitness_history)
    generations = np.arange(fitness_history.shape[0])
        
    plt.plot(generations, fitness_history[:, 0], marker='o', label='Objective 1')
    plt.plot(generations, fitness_history[:, 1], marker='o', label='Objective 2')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness History Over Generations')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def calculate_domination(fitness):
    domination_count = {i: 0 for i in range(fitness.shape[0])}
    dominated_solutions = {i: [] for i in range(fitness.shape[0])}
    for i in range(fitness.shape[0]):
        for j in range(fitness.shape[0]):
            if i != j:
                if np.all(fitness[i] >= fitness[j]) and np.any(fitness[i] > fitness[j]):
                    domination_count[j] += 1
                    dominated_solutions[j].append(i)
                elif np.all(fitness[j] >= fitness[i]) and np.any(fitness[j] > fitness[i]):
                    domination_count[i] += 1
                    dominated_solutions[i].append(j)
    
    return domination_count, dominated_solutions

def calculate_domination_matrix(fitness):
    num_solutions = fitness.shape[0]
    domination_matrix = np.zeros((num_solutions, num_solutions), dtype=int)
    for i in range(num_solutions):
        for j in range(num_solutions):
            if i != j:
                if np.all(fitness[i] >= fitness[j]) and np.any(fitness[i] > fitness[j]):
                    domination_matrix[j, i] = 1
    return domination_matrix



#function to get the pareto fronts
def get_pareto_fronts(domination_matrix):
    pareto_fronts = []
    num_solutions = domination_matrix.shape[0]
    original_indices = np.arange(num_solutions)
    
    while num_solutions > 0:
        current_front = np.where(domination_matrix.sum(axis=1) == 0)[0]
        pareto_fronts.append(original_indices[current_front].tolist())
        # Remove the rows and columns corresponding to the current front
        domination_matrix = np.delete(domination_matrix, current_front, axis=0)
        domination_matrix = np.delete(domination_matrix, current_front, axis=1)
        original_indices = np.delete(original_indices, current_front)
        num_solutions = domination_matrix.shape[0]
    return pareto_fronts

def calculate_crowding_distance(fitness, pareto_fronts):
    crowding_distance = np.zeros(fitness.shape[0])
    for front in pareto_fronts:
        front = np.array(front)
        for objective in range(fitness.shape[1]):
            #here [::-1] for maximasiation problems, do nothing for min problems
            sorted_front = front[np.argsort(fitness[front, objective])[::-1]]
            crowding_distance[sorted_front[0]] = np.inf
            crowding_distance[sorted_front[-1]] = np.inf
            for i in range(1, len(sorted_front)-1):
                crowding_distance[sorted_front[i]] += (fitness[sorted_front[i+1], objective]-fitness[sorted_front[i-1], objective])/(np.max(fitness[:, objective])-np.min(fitness[:, objective]))
    return crowding_distance

def select_top_n_by_crowding_distance(population, crowding_distance, n):
    top_indices = np.argsort(crowding_distance)[::-1][:n]
    return population[top_indices]

def nsga_ii(population_size, items, max_capacity, num_objectives, generations,crossover_rate,mutate_rate, num_items):
        population = initialize_population(population_size, num_items)
        # print("Fitness:\n", fitness)
        fitness_history = []
        solution_history = []
        for generation in range(generations):
                fitness = fitness_function(population, items, max_capacity, num_objectives)
                # print("Fitness:\n", fitness)
                normalized_fitness = normalize_fitness(fitness)
                # print("Normalized Fitness:\n", normalized_fitness)

                weighted_fitness = normalized_fitness[0] + normalized_fitness[1]
                best_index = np.argmax(weighted_fitness)
                fitness_history.append(fitness[best_index])
                solution_history.append(population[best_index])

                selected_parents = tournament_selection(population, normalized_fitness, 3)
                # print("Selected Parents:\n", selected_parents)

                children = crossover(selected_parents, crossover_rate)
                # children = two_point_crossover(selected_parents, crossover_rate)
                # print("Children after Crossover:\n", children)

                mutated_children = mutate(children, mutate_rate)
                # print("Mutated Children:\n", mutated_children)

                combined_population = np.concatenate((population, mutated_children), axis=0)
                # print("Combined Population:\n", combined_population)

                fitness = fitness_function(combined_population, items, max_capacity, num_objectives)
                # print("Fitness:\n", fitness)

                domination_matrix = calculate_domination_matrix(fitness)
                # print("Domination Matrix:\n", domination_matrix)

                pareto_fronts = get_pareto_fronts(domination_matrix)
                # print("Pareto Fronts:\n", pareto_fronts)

                # best_Pareto_fitness = fitness[pareto_fronts[0]]
                # # print("Best Pareto:\n", best_Pareto_fitness)
                # normalized_fitness = normalize_fitness(best_Pareto_fitness)
                # print("Normalized Fitness:\n", normalized_fitness)
                # weighted_fitness = normalized_fitness[0] + normalized_fitness[1]
                # best_index = np.argmax(weighted_fitness)
                # fitness_history.append(best_Pareto_fitness[best_index])
                # solution_history.append(combined_population[pareto_fronts[0]][best_index])

                # if generation %50==0 or generation==1:
                        # plot_objectives(combined_population, items, max_capacity, pareto_fronts)

                crowding_distance = calculate_crowding_distance(fitness, pareto_fronts)
                # print("Crowding Distance:\n", crowding_distance)

                population = select_top_n_by_crowding_distance(combined_population, crowding_distance, population_size)              
                # print("top population:\n", population)
        return fitness_history, solution_history