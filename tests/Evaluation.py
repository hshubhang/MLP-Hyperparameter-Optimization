import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
from src.model import MLP_model  
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Add print statements to check progress
print("Loading bayes_result...")
bayes_result = joblib.load('bayes_result.pkl')
print("Loaded bayes_result.")

print("Loading Xtr and ytr...")
Xtr = joblib.load('Xtr.pkl')
ytr = joblib.load('ytr.pkl')
print("Loaded Xtr and ytr.")

# Check the shapes of the data
print(f"Xtr shape: {Xtr.shape}, ytr shape: {ytr.shape}")



# bayes_result = joblib.load('bayes_result.pkl')  # Load the saved Bayesian result
# Xtr = joblib.load('Xtr.pkl')  # Load training data
# ytr = joblib.load('ytr.pkl')  # Load training labels

best_model = bayes_result.best_estimator_

print("Starting cross-validation...")
precision_scores = cross_val_score(best_model, Xtr, ytr, cv=10, scoring='precision')
recall_scores = cross_val_score(best_model, Xtr, ytr, cv=10, scoring='recall')
f1_scores = cross_val_score(best_model, Xtr, ytr, cv=10, scoring='f1')

print("Precision: %0.3f (+/- %0.3f)" % (precision_scores.mean(), precision_scores.std() * 2))
print("Recall: %0.3f (+/- %0.3f)" % (recall_scores.mean(), recall_scores.std() * 2))
print("F1 Score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))