import sys
import os

# Dynamically add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import joblib
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from src.data_processing import data_processing
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

Xtr, Xts, ytr, yts = data_processing() 

def MLP_model(n_layers=1, n_units=16, activation_fn='relu', optimizer='SGD', learning_rate=0.01):
    model = Sequential()
    model.add(Dense(units=n_units, activation=activation_fn, input_dim=Xtr.shape[1]))
    for i in range(n_layers - 1):
        model.add(Dense(units=n_units, activation=activation_fn))
    model.add(Dense(units=1, activation='sigmoid'))
    if optimizer == 'SGD':
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'Adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


#Bayesian Optimization and Grid Search
param_space = {'n_layers': [1,2,3],
               'n_units': [16, 128],
               'activation_fn': ['relu', 'tanh'],
               'optimizer': ['SGD', 'Adam'],
               'learning_rate': [0.001, 0.01, 0.1]}
model = KerasClassifier(build_fn=MLP_model, verbose=0)

bayes = BayesSearchCV(estimator=model, search_spaces=param_space, n_iter=50, cv=10, n_jobs=-1)
bayes_result = bayes.fit(Xtr, ytr)
print("Best parameters: ", bayes_result.best_params_)
print("Accuracy: ", bayes_result.best_score_)

grid = GridSearchCV(estimator=model, param_grid=param_space, cv=10)
grid_result = grid.fit(Xtr, ytr)
print("Best parameters: ", grid_result.best_params_)
print("Accuracy: ", grid_result.best_score_)


best_model = bayes_result.best_estimator_

print("Starting cross-validation...")
precision_scores = cross_val_score(best_model, Xtr, ytr, cv=10, scoring='precision', n_jobs=-1)
recall_scores = cross_val_score(best_model, Xtr, ytr, cv=10, scoring='recall', n_jobs=-1)
f1_scores = cross_val_score(best_model, Xtr, ytr, cv=10, scoring='f1',n_jobs=-1)

print("Precision: %0.3f (+/- %0.3f)" % (precision_scores.mean(), precision_scores.std() * 2))
print("Recall: %0.3f (+/- %0.3f)" % (recall_scores.mean(), recall_scores.std() * 2))
print("F1 Score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))

#Plotting the Scores
mean_precision = np.mean(precision_scores)
std_precision = np.std(precision_scores)
mean_recall = np.mean(recall_scores)
std_recall = np.std(recall_scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
labels = ['Precision', 'Recall', 'F1 Score']
means = [mean_precision, mean_recall, mean_f1]
stds = [std_precision, std_recall, std_f1]
color = 'blue'
capsize = 10
plt.errorbar(labels, means, yerr=stds, fmt='o', color=color, ecolor=color, capsize=capsize)
plt.title('Error Bars of Precision, Recall, and F1 Score')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.show()