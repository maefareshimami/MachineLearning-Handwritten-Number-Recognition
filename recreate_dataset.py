import random as rd
import numpy as np
from sklearn.datasets import fetch_openml
from PIL import Image as Img

import constants as cst


def recreateDataset()->None:
    """Recreate the dataset with image to make the program more flexible"""
    ##  Dataset ##
    dataset = fetch_openml("mnist_784", version = 1)
    sample_x = rd.sample([i for i in range(0, cst.NB_DATA)], cst.NB_DATA_X)     # I don't choose all data because of the compute time
    with open("dataset\\dataset_true_values.txt", "w", encoding = "utf-8") as f:
        pass
    for i, j in enumerate(sample_x):
        x_i = dataset.data.iloc[j].to_numpy()
        img_array = x_i.reshape(cst.WIDTH, cst.HEIGHT)     # 28 * 28 = 784 attributes of x[j]
        img = Img.fromarray(img_array.astype(np.uint8))
        img.save(f"dataset\\img_{i}.jpg")
        with open("dataset\\dataset_true_values.txt", "a", encoding = "utf-8") as f:
            f.write(f"{dataset.target[j]}\n")     # Value of x[j] between 0 and 9

    ##  Training dataset  ##
    sample_a = rd.sample([i for i in range(0, cst.NB_DATA_X)], cst.NB_DATA_A)
    with open("training_dataset\\dataset_training_index.txt", "w", encoding = "utf-8") as f:
        for elt in sample_a:
            f.write(f"{elt}\n")
    return None