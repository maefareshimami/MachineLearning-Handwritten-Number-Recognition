import numpy as np
import random as rd
import time

import constants as cst
import algorithms as algo


def performances(x:np.array, dict_x:dict, sample_a:list, a:np.array, dict_a:dict, k:int)->(np.array, np.array, float, float, float, list, list, list, list, list):
    """Show the confusion matrix for a k"""
    t1 = time.time()     # To know the time to compute the confusion matrix
    nb_data_x, _ = x.shape
    confusion_matrix = np.zeros((cst.NB_LABELS, cst.NB_LABELS), dtype = int)
    count = 0
    list_i = []
    while count < cst.NB_DATA_TEST:     # Select data which aren't used in the training dataset
        i = rd.randint(0, nb_data_x - 1)
        if i not in sample_a and i not in list_i:
            new_x = x[i, :]
            true_label = dict_x[i]
            predicted_label = algo.kNN(a, dict_a, k, new_x)
            confusion_matrix[int(true_label), int(predicted_label)] += 1     # Rows: True value ; Columns: Value estimated
            list_i.append(i)
            count += 1
    t2 = time.time()     # The compute time of performances isn't important
    normalized_matrix = confusionMatrixNormalized(confusion_matrix)
    acc = accuracy(confusion_matrix)
    list_prev = prevalence(confusion_matrix)
    list_preci = precision(confusion_matrix)
    lsit_recall = recall(confusion_matrix)
    list_f1_score, list_fbeta_score = fScore(confusion_matrix, cst.BETA)
    mcc_result = mcc(confusion_matrix)
    return confusion_matrix, normalized_matrix, t2 - t1, acc, mcc_result, list_prev, list_preci, lsit_recall, list_f1_score, list_fbeta_score

def support(conf_matrix:np.array)->list:
    """Compute the number of data on each row"""
    nb_labels = conf_matrix.shape[0]
    list_supp = []
    for i in range(0, nb_labels):
        list_supp.append(sum(conf_matrix[i, :]))     # Sum of the row i
    return list_supp

def confusionMatrixNormalized(conf_matrix:np.array)->np.array:
    """Compute the mormalized matrix from a matrix"""
    nb_labels = conf_matrix.shape[0]
    list_supp = support(conf_matrix)     # Number of data on each row
    normalized_matrix = np.zeros((nb_labels, nb_labels), dtype = float)
    for i in range(0, nb_labels):
        for j in range(0, nb_labels):
            normalized_matrix[i, j] = round(conf_matrix[i, j] / list_supp[i], 4)
    return normalized_matrix

def accuracy(conf_matrix:np.array)->float:
    """Compute the accuracy: ratio of right data for any value"""
    nb_labels = conf_matrix.shape[0]
    sum_total = 0.0
    diagonal = 0.0
    for i in range(0, nb_labels):
        for j in range(0, nb_labels):
            sum_total += conf_matrix[i][j]
            if i == j:
                diagonal += conf_matrix[i][j]
    return round(diagonal / sum_total * 100.0, 2)

def mcc(conf_matrix:np.array)->float:
    """Compute the Matthews correlation coefficient (MCC): useful if classes have very different sizes"""
    nb_labels = conf_matrix.shape[0]
    correct_predictions = 0
    nb_data_true_class = []     # Elements of a same row
    nb_data_predicted_class = []     # Elements of a same column
    for i in range(0, nb_labels):
        nb_data_row = 0
        nb_data_columns = 0
        for j in range(0, nb_labels):
            if i == j:
                correct_predictions += conf_matrix[i, j]
            nb_data_row += conf_matrix[i, j]
            nb_data_columns += conf_matrix[j, i]
        nb_data_true_class.append(nb_data_row)     # List of sums by row
        nb_data_predicted_class.append(nb_data_columns)     # List of sums by column
    true_predicted = 0
    true_true = 0
    predicted_predicted = 0
    for i in range(0, nb_labels):
        true_predicted += nb_data_true_class[i] * nb_data_predicted_class[i]
        true_true += nb_data_true_class[i] * nb_data_true_class[i]     # Multiplication is more efficient than power
        predicted_predicted += nb_data_predicted_class[i] * nb_data_predicted_class[i]
    numerator_mcc = correct_predictions * cst.NB_DATA_TEST - true_predicted
    denominator_mcc = np.sqrt(cst.NB_DATA_TEST * cst.NB_DATA_TEST - predicted_predicted) * np.sqrt(cst.NB_DATA_TEST * cst.NB_DATA_TEST - true_true)
    return numerator_mcc / denominator_mcc

def prevalence(conf_matrix:np.array)->list:
    """Compute the prevalence: ratio of a label among all labels (so by row)"""
    nb_labels = conf_matrix.shape[0]
    list_prev = []     # List of all prevalences
    for i in range(0, nb_labels):
        list_prev.append(conf_matrix[i, i] / cst.NB_DATA_TEST)     # Prevalence of line i
    return list_prev

def precision(conf_matrix:np.array)->list:
    """Compute the precision: ratio of real right elements among right elements (so by column)"""
    nb_labels = conf_matrix.shape[0]
    list_preci = []     # List of all precisions
    for i in range(0, nb_labels):
        total_column = 0
        for j in range(0, nb_labels):
            total_column += conf_matrix[j, i]     # Indexes i and j are reversed browse columns
        list_preci.append(conf_matrix[i, i] / total_column)
    return list_preci

def recall(conf_matrix:np.array)->list:
    """Compute the recall: ratio of right elements among right elements excepted (so by row)"""
    nb_labels = conf_matrix.shape[0]
    list_recall = []     # List of all precisions
    for i in range(0, nb_labels):
        total_row = 0
        for j in range(0, nb_labels):
            total_row += conf_matrix[i, j]
        list_recall.append(conf_matrix[i, i] / total_row)
    return list_recall

def fScore(conf_matrix:np.array, beta:float)->(list, list):
    """Compute the F1-score and the Fbeta-score: harmonic mean of precision and recall, with or without the weight beta"""
    nb_labels = conf_matrix.shape[0]
    list_preci = precision(conf_matrix)
    list_recall = recall(conf_matrix)
    list_f1_score = []
    list_fbeta_score = []
    for i in range(0, nb_labels):
        p = list_preci[i]
        r = list_recall[i]
        list_f1_score.append(2 *(p * r) / (p + r))
        list_fbeta_score.append((1 + beta ** 2) *(p * r) / (beta **2 * p + r))
    return list_f1_score, list_fbeta_score


def optimalK(x:np.array, dict_x:dict, sample_a:list, a:np.array, dict_a:dict, nb_loops:int, nb_data_x:int)->(int, float, float):
    """Loop to compute the k optimal"""
    t1 = time.time()
    optimal_accuracy = 0.0
    for i in range(1, nb_data_x):
        total_sum = 0.0
        for _ in range(nb_loops):
            _, accuracy, _ = performances(x, dict_x, sample_a, a, dict_a, i)
            total_sum += accuracy
        average = total_sum / nb_loops
        if average >= optimal_accuracy:
            optimal_accuracy = average
            optimal_k = i
            print(f"Last maximum accuracy: {optimal_accuracy} %")     # Give some information because the compute could be very long...
            print(f"Last optimal k: {optimal_k}\n")
    t2 = time.time()
    return optimal_k, optimal_accuracy, t2 - t1