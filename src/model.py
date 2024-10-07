import sys
import os

# Dynamically add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import joblib
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.decomposition import TruncatedSVD
from src.data_processing import data_processing

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

joblib.dump(bayes_result, 'bayes_result.pkl')
joblib.dump(Xtr, 'Xtr.pkl')
joblib.dump(ytr, 'ytr.pkl')