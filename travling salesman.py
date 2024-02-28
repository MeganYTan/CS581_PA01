import numpy as np
import pandas as pd
from itertools import combinations

# Example 2D array of locations: [name, x_coordinate, y_coordinate]
locations_df = pd.read_csv("campus.csv", header=None)
locations_df.columns = ["Name", "X", "Y"]
# locations = [
#     ["Location A", 0, 0],
#     ["Location B", 2, 3],
#     ["Location C", 5, 2],
#     ["Location D", 6, 6],
#     ["Location E", 8, 3],
# ]


# Calculate the Euclidean distance between two points
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Calculate the Euclidean distance between two points
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Compute distance matrix from DataFrame
def compute_distance_matrix(locations_df):
    n = len(locations_df)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = distance(
                    locations_df.iloc[i][['X', 'Y']].values,
                    locations_df.iloc[j][['X', 'Y']].values)
    return distance_matrix


# Adapted Held-Karp algorithm for DataFrame
def held_karp(distance_matrix):
    n = len(distance_matrix)
    C = {}

    for k in range(1, n):
        C[(1 << k, k)] = (distance_matrix[0][k], 0)

    for subset_size in range(2, n):
        for subset in combinations(range(1, n), subset_size):
            bits = 0
            for bit in subset:
                bits |= 1 << bit
            for k in subset:
                prev = bits & ~(1 << k)
                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + distance_matrix[m][k], m))
                C[(bits, k)] = min(res)

    bits = (2 ** n - 1) - 1
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + distance_matrix[k][0], k))
    opt, parent = min(res)

    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits
    path.append(0)

    return opt, list(reversed(path))


distance_matrix = compute_distance_matrix(locations_df)
optimal_length, path_indices = held_karp(distance_matrix)

# Map indices to location names
optimal_path_names = [locations_df.iloc[p]['Name'] for p in path_indices]

# optimal_length, optimal_path_names
print(optimal_length)
print(optimal_path_names)