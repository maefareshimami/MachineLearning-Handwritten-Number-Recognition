from sklearn.datasets import fetch_openml


dataset = fetch_openml("mnist_784", version = 1)
NB_DATA = len(dataset.data)     # 70 000 data
NB_DATA_X = 1000     # Less than 70 000 to reduce the compute time
TRAINING_DATASET_SIZE = 0.75     # Percentage of dataset used for the training dataset
NB_DATA_A = round(TRAINING_DATASET_SIZE * NB_DATA_X)
NB_DATA_TEST = round((1 - TRAINING_DATASET_SIZE) * NB_DATA_X)

NB_LABELS = 10     # I work with numbers between 0 and 9

WIDTH = 28     # Size of the images
HEIGHT = 28

EPSILON = 1e-4     # Minimum value to consider the label of an image is the real value
BETA = 0.5     # For the Fbeta-score