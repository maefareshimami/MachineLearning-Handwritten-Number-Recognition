import numpy as np

import constants as cst


def printResultsOnTxt(confusion_matrix:np.array, normalized_matrix:np.array, compute_time:float, acc:float, mcc_result:float, list_prev:list, list_preci:list, list_recall:list, list_f1_score:list, list_fbeta_score:list)->None:
    with open("results.txt", "w", encoding = "utf-8") as f:
        f.write(f"Dataset: {cst.NB_DATA_X} values\n\n")
        f.write(f"Training dataset ({cst.TRAINING_DATASET_SIZE * 100} %): {cst.NB_DATA_A} values\n\n")

        f.write(f"Confusion Matrix:\n{confusion_matrix}\n\n")
        f.write(f"Normalized Confusion Matrix:\n{normalized_matrix}\n")
        f.write(f"Compute time: {round(compute_time, 2)} s\n\n")

        f.write(f"Accuracy: {round(acc, 4)} %\n")
        f.write(f"MCC: {round(mcc_result, 4)}\n\n")

        f.write(f"{"Classification":<10}\t{"Prevalence":<10}\t{"Precision":<10}\t{"Recall":<10}\t{"F1-score":<10}\t{f"Fbeta-score, beta = {cst.BETA}":<10}\n\n")
        for i in range(0, cst.NB_LABELS):
            f.write(f"{i:<10}\t{round(list_prev[i], 4):<10}\t{round(list_preci[i], 4):<10}\t{round(list_recall[i], 4):<10}\t{round(list_f1_score[i], 4):<10}\t{round(list_fbeta_score[i], 4):<10}\n")