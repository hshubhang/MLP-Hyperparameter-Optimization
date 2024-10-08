# MLP-Hyperparameter-Optimization

This repository contains an implementation of hyperparameter optimization for a MultiLayer Perceptron (MLP) model using `scikit-optimize` and `GridSearchCV`. The project showcases how to fine-tune the MLP model to achieve the best performance on a given dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
- [File Structure](#file-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to demonstrate the process of hyperparameter optimization for a MultiLayer Perceptron (MLP) model using Bayesian Optimization (via `scikit-optimize`) and Grid Search. The dataset used for training is the Breast Cancer dataset from `scikit-learn`.

## Technologies Used
- Python
- TensorFlow
- Keras
- scikit-learn
- scikit-optimize
- pandas
- numpy

## Installation

### 1. Clone the Repository
First, clone this repository to your local machine:
```bash
git clone https://github.com/your-username/MLP-Hyperparameter-Optimization.git
cd MLP-Hyperparameter-Optimization
```
## 2. Environment creation and Installation of dependenices
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install the required dependencies using the requirements.txt file
```bash
pip install -r requirements.txt
```
### 3. Train the model

Training the Model

To train the model and perform hyperparameter optimization using Bayesian Optimization or Grid Search, run the following command:
```bash
python src/model.py
```
The script performs:

	•	Bayesian Optimization with BayesSearchCV
	•	Grid Search using GridSearchCV
    •   Perform cross-validation
	•	Print metrics such as precision, recall, and F1 score.


## Results

After running the hyperparameter optimization, the best hyperparameters will be displayed, along with evaluation metrics like accuracy, precision, recall, and F1 score

```bash
Best parameters: {'n_layers': 2, 'n_units': 128, 'activation_fn': 'relu', 'optimizer': 'Adam', 'learning_rate': 0.01}
Accuracy: 0.97
```


