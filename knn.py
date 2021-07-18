
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix
from preprocess import *
import lightgbm as lgb


def run_knn(X,y):

    cv_outer = KFold(n_splits=10, shuffle=False, random_state=1)

    for x_train_indecies, y_train_indecies in  cv_outer.split(X):
        X_train = X
