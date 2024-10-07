from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from src import bayes_result, Xtr, ytrx

best_model = bayes_result.best_estimator_

precision_scores = cross_val_score(best_model, Xtr, ytr, cv=10, scoring='precision')
recall_scores = cross_val_score(best_model, Xtr, ytr, cv=10, scoring='recall')
f1_scores = cross_val_score(best_model, Xtr, ytr, cv=10, scoring='f1')

print("Precision: %0.3f (+/- %0.3f)" % (precision_scores.mean(), precision_scores.std() * 2))
print("Recall: %0.3f (+/- %0.3f)" % (recall_scores.mean(), recall_scores.std() * 2))
print("F1 Score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))