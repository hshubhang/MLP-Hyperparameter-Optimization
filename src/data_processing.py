
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

def data_processing(test_size=0.2, random_state=50):
#Loading and Normalizing the data
    data = load_breast_cancer()
    X, y = data.data, data.target

    Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=test_size, random_state=random_state)

    mean = Xtr.mean(axis=0)
    stdiv = Xtr.std(axis=0)
    Xtr = (Xtr - mean) / stdiv
    Xts = (Xts - mean) / stdiv

    return Xtr, Xts, ytr, yts