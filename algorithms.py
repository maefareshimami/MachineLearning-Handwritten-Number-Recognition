import numpy as np

import constants as cst


def kNN(a:np.array, dict_a:dict, k:int, x:np.array)->int:
    """k-Nearest Neighbors algorithm"""
    list_data_distance = []
    nb_data_a, nb_attributes = a.shape
    for i in range(0, nb_data_a):
        distance2 = 0.0
        for j in range(0, nb_attributes):
            distance2 += (a[i, j] - x[j]) * (a[i, j] - x[j])     # Multiplication is more efficient than power
        distance = np.sqrt(distance2)
        list_data_distance.append((dict_a[i], distance))
    list_data_distance = sorted(list_data_distance, key=lambda list_data_distance: list_data_distance[1])[0:k]     # Sort by distance, the second element of the tuple
    return mainLabel(list_data_distance, cst.EPSILON)

def mainLabel(list_data_distance:list, epsilon:float)->int:
    """Return the label which is the most represented for the k first neighbors"""
    if list_data_distance[0][1] <= epsilon:
        return list_data_distance[0][0]
    dict_frequency = {}
    frequency_max = 0
    for elt in list_data_distance:
        nearest_elt = elt[0]
        if nearest_elt in dict_frequency:
            dict_frequency[nearest_elt] += 1
        else:
            dict_frequency[nearest_elt] = 1
        if dict_frequency[nearest_elt] >= frequency_max:
            frequency_max_key = nearest_elt
            frequency_max = dict_frequency[frequency_max_key]
    return frequency_max_key