import copy
import random
import sys
import numpy as np
import pandas as pd
from random import randrange, uniform
import math
import timeit


# import csv


def print_cl_error():
    print("Not enough or too many input arguments.")
    sys.exit()


def print_result_header():
    print("Tan, Megan, A20527707 solution:")
    print(f"Initial state: {initial_state}")
    print()
    if algorithm == 1:
        # simulated
        print("Simulated Annealing: ")
        print(f"Command Line Parameters: temperature = {p1}, alpha = {p2}")
    elif algorithm == 2:
        # genetic
        print("Genetic Algorithm: ")
        print(f"Command Line Parameters: number of iterations = {p1}, mutation probability = {p2}")


def print_result_body():
    print("Initial solution: ", end="")
    if algorithm == 1:
        print(array_to_labels(initial_solution))
    elif algorithm == 2:
        print([array_to_labels(i) for i in initial_solution])
    print("Final solution: ", end="")
    print(array_to_labels(final_solution))
    print(f"Number of iterations {number_of_iterations}")
    print("Total time taken: {}".format(elapsed_time_in_sec))
    print(f"Complete path cost: {solution_cost}")


def array_to_labels(array):
    # array is state of numbers eg 0,1,2
    labels = [matrix.iloc[i]['State'] for i in array]
    # labels.append(matrix.iloc[0]['State'])
    return labels


def read_file_into_matrix():
    # read file and convert to matrix
    # input file is state, X_coord, Y_coord
    # try:
    #     file = open(input_file_name,mode="r", encoding='utf-8-sig')
    # except:
    #     print("Error reading file")
    #     sys.exit()
    # matrix = []
    # for line in file:
    #     # print(line)
    #     replaced_line = line.replace('\n', '')
    #     line_arr = replaced_line.split(',')
    #     # line_arr = [int(x) for x in line_arr]
    #     matrix.append(line_arr)
    # print(matrix)
    matrix = pd.read_csv(input_file_name, header=None)
    matrix.columns = ["State", "X", "Y"]
    # print(matrix)
    return matrix


def write_matrix_into_file():
    suffix = ""
    if algorithm == 1:
        # simulated
        suffix = "SA"
    else:
        # genetic
        suffix = "GA"
    output_file_name = input_file_name[0:-4] + "_" + suffix + ".txt"
    # get states
    solution_labels = array_to_labels(final_solution)
    # first line should be total path cost
    solution_labels.insert(0, solution_cost)
    with open(output_file_name, 'w') as f:
        for line in solution_labels:
            f.write(f"{line}\n")


def create_random_solution():
    hash_table = np.zeros(number_of_states).tolist()
    hash_table[0] = 1
    solution = [0]
    while len(solution) < number_of_states:
        # generate random integer
        next_spot = randrange(1, number_of_states)
        if hash_table[next_spot] == 0:
            # has not yet been added to solution
            solution.insert(0, next_spot)
            hash_table[next_spot] = 1
    return solution


def calculate_distance(array):
    # make illegal solutions unviable
    if len(array) != len(set(array)):
        return -1
    total_distance = 0
    for i in range(1, number_of_states):
        x_distance_squared = (matrix.iloc[array[i - 1]]['X'] + matrix.iloc[array[i]]['X']) ** 2
        y_distance_squared = (matrix.iloc[array[i - 1]]['Y'] + matrix.iloc[array[i]]['Y']) ** 2
        distance = math.sqrt(x_distance_squared + y_distance_squared)
        total_distance += distance
    x_distance_squared = (matrix.iloc[array[number_of_states - 1]]['X'] + matrix.iloc[array[0]]['X']) ** 2
    y_distance_squared = (matrix.iloc[array[number_of_states - 1]]['Y'] + matrix.iloc[array[0]]['Y']) ** 2
    distance = math.sqrt(x_distance_squared + y_distance_squared)
    total_distance += distance
    return total_distance


def swap_two_spots(solution):
    # swap 2 spots
    new_solution = copy.deepcopy(solution)
    # swap from 0 to number_of_states -1 so INITIAL_STATE always last label
    spot1 = randrange(0, number_of_states - 1)
    spot2 = randrange(0, number_of_states - 1)
    # in case both spots are the same
    while spot2 == spot1:
        spot2 = randrange(number_of_states)
    new_solution[spot1], new_solution[spot2] = new_solution[spot2], new_solution[spot1]
    return new_solution


def simulated_annealing():
    temp = float(p1)
    alpha = float(p2)
    # initial solution is random solution
    initial_solution = create_random_solution()
    # copy created so initial_solution unchanged (for comparison and printing purpose)
    solution = copy.deepcopy(initial_solution)
    curr_solution_distance = calculate_distance(solution)
    # count number of iterations
    count = 0
    while temp > 0:
        count += 1
        # select random successor
        new_solution = swap_two_spots(solution)
        # check if new solutiion better than current one
        new_solution_distance = calculate_distance(new_solution)
        delta = curr_solution_distance - new_solution_distance
        if delta > 0:
            # accept always as new solution is better
            solution = new_solution
            curr_solution_distance = new_solution_distance
        else:
            # accept with probability
            probability_accept = math.exp(delta / temp)
            generated_probability = uniform(0, 1)
            if generated_probability < probability_accept:
                solution = new_solution
                curr_solution_distance = new_solution_distance
        temp = math.exp(-count * alpha) * temp
    return initial_solution, solution, count, curr_solution_distance


def genetic_algorithm():
    # HELPER FUNCTIONS FOR GENETIC ALGORITHM
    def select_parent():
        return population[np.random.choice(len(population), p=population_fitness_ratio)]

    def mutate(individual, mutation_probability):
        probability_generated_for_mutation = random.uniform(0, 1)
        if probability_generated_for_mutation < mutation_probability:
            swap_two_spots(individual)
        return individual

    # this is for a separate experiment using a smarter crossover strategy
    def mate_with_orderedcrossover(parent_1, parent_2):
        new_individual = [None] * number_of_states
        # # copy values from parent 1
        for i in range(crossover_point_1, crossover_point_2):
            # select from par 1
            new_individual[i] = parent_1[i]
        # missing values get in order of parent 2
        par_2_pointer = 0
        for i in range(0, number_of_states):
            if new_individual[i] is None:
                # find next par 2 val not in new individual and copy from par 2
                while parent_2[par_2_pointer] in new_individual:
                    par_2_pointer += 1
                new_individual[i] = parent_2[par_2_pointer]
        return new_individual

    def mate(parent_1, parent_2):
        new_individual = [None] * number_of_states
        for i in range(crossover_point_1):
            new_individual[i] = parent_1[i]
        for i in range(crossover_point_1, crossover_point_2):
            new_individual[i] = parent_2[i]
        for i in range(crossover_point_2, number_of_states):
            new_individual[i] = parent_1[i]
        return new_individual

    # Genetic Algorithm Setup
    population_size = 10
    k = float(p1)
    mutation_probability = float(p2)
    # Genetic Algorithm
    # Step 0: create random population
    population = [create_random_solution() for _ in range(0, population_size)]
    # save initial solution for printing purpose
    initial_solution = copy.deepcopy(population)
    # count is set to -1 because first part of loop is run one more time to get fitness value
    count = -1
    while True:
        # Step 1: calculate fitness
        population_fitness = [calculate_distance(i) for i in population]
        # Step 1a: get fitness ratio
        # we are only taking fitness values over -1 since that is the illegal value
        min_index, min_fitness = min([(index, value) for index, value in enumerate(population_fitness) if value > -1])
        max_fitness = max(population_fitness)
        if count >= k:
            # solution is population with min fitness
            solution = population[min_index]
            break
        rescaled_population_fitness = [max_fitness - individual_fitness for individual_fitness in
                                       population_fitness]
        sum_of_all_fitness = sum(rescaled_population_fitness)
        population_fitness_ratio = [rescaled_individual_fitness / sum_of_all_fitness for rescaled_individual_fitness in
                                    rescaled_population_fitness]
        # Steps 2,3,4: select, mate, mutate - roulette wheel. 2 point crossover. swap 2 points
        crossover_point_1 = math.floor(0.25 * number_of_states)
        crossover_point_2 = math.ceil(0.5 * number_of_states)
        new_population = []
        # add lowest fitness parent to new population to ensure there is at least 1 legal population
        new_population.append(population[min_index])
        while len(new_population) < population_size:
            # Step 2: Select 2 parent individuals with roulette wheel selection
            selected_individual_1 = select_parent()
            selected_individual_2 = select_parent()
            # Step 3: Mate with 2 point crossover
            new_individual = mate(selected_individual_1, selected_individual_2)
            # Step 4: mutate with mutation probability by swapping 2 points
            new_individual = mutate(new_individual, mutation_probability)
            # Step 5: add new individual to new population
            new_population.append(new_individual)
        # Step 6: replace population with new population
        population = new_population
        count += 1
    # return initial_solution, solution, count, curr_solution_distance
    return initial_solution, solution, count, min_fitness


if __name__ == '__main__':
    input_file_name = "campus.csv"
    algorithm = 2  # 1 simulated, 2 genetic
    p1 = 10  # if simulated temperature, if genetic iterations
    p2 = 0.1  # is simulated temperature cooling schedule a, if genetic mutation probability value

    # if ran from command line
    if sys.stdin.isatty():
        cl_arguments = sys.argv[1:]
        if len(cl_arguments) != 4:
            print_cl_error()
        else:
            input_file_name = cl_arguments[0]
            algorithm = int(cl_arguments[1])
            p1 = cl_arguments[2]
            p2 = cl_arguments[3]
    # read file
    matrix = read_file_into_matrix()
    # set up needed variables
    initial_state = matrix.iloc[0]['State']
    number_of_states = len(matrix)
    # print result header
    print_result_header()
    # run the algorithm and also time it
    time_start = timeit.default_timer()
    if algorithm == 1:
        initial_solution, final_solution, number_of_iterations, solution_cost = simulated_annealing()
    elif algorithm == 2:
        initial_solution, final_solution, number_of_iterations, solution_cost = genetic_algorithm()
    else:
        print("Error - No such algorithm")
        exit()
    time_end = timeit.default_timer()
    elapsed_time_in_sec = time_end - time_start
    # print results
    print_result_body()
    # write results into file
    write_matrix_into_file()
