import re

import os
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split

import ml_dataset_loader.datasets as data_loader

test_size = 0.2
train_size = 0.6
validation_size = 0.2
random_seed = 7
# num_rounds = 10000
num_rounds = 15
early_stopping_rounds = 100

common_param = {'eta': 0.05, 'silent': 1, 'tree_method': 'gpu_hist'}
optimizer_param = [
    {'name': 'Baseline', 'optimizer': 'default_optimizer'},
    {'name': 'Momentum', 'optimizer': 'momentum_optimizer', 'momentum': 0.3},
    {'name': 'Nesterov', 'optimizer': 'nesterov_optimizer', 'momentum': 0.3},
    {'name': 'AddSign', 'optimizer': 'add_sign_optimizer', 'as_alpha': 1.5},
    {'name': 'PowerSign', 'optimizer': 'power_sign_optimizer', 'base': 2.0},
]


class Experiment:
    def __init__(self, name, objective, metric, load_data):
        self.name = name
        self.objective = objective
        self.metric = metric
        self.load_data = load_data

    def run(self, num_rows=None):
        # Create train/test/validation sets
        # Don't use full airline dataset
        if self.name == "Airline" and num_rows is None:
            num_rows = 10000000
        X, y = self.load_data(num_rows)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            random_state=random_seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                          y_train,
                                                          test_size=validation_size / train_size,
                                                          random_state=random_seed)
        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test, y_test)
        dval = xgb.DMatrix(X_val, y_val)

        data = {"metric": self.metric, "results": {}}
        for opt_param in optimizer_param:
            param = common_param.copy()
            param.update(opt_param)
            param["objective"] = self.objective
            param["eval_metric"] = self.metric

            if self.objective == "reg:linear":
                param["base_score"] = np.average(y_train)
            if self.objective == "multi:softmax":
                param["num_class"] = np.max(y_train) + 1
            if self.objective == "binary:logistic":
                pos = np.sum(y_train)
                param["scale_pos_weight"] = (len(y_train) - pos) / pos
                print(param["scale_pos_weight"])

            res = {}
            bst = xgb.train(param, dtrain, num_rounds,
                            evals=[(dtrain, "train"), (dtest, "test"), (dval, "val")],
                            early_stopping_rounds=early_stopping_rounds, evals_result=res)
            data["results"][param["name"]] = res['test'][self.metric][:bst.best_iteration + 1]
            del bst

        # Save data
        if not os.path.exists("data"):
            os.makedirs("data")
        snake_name = re.sub(' ', '_', self.name).lower()
        f = open("data/" + snake_name + "_data.pkl", "wb")
        pickle.dump({self.name: data}, f)
        f.close()


experiments = [
    Experiment("Synthetic classification", "binary:logistic", "error", data_loader.get_synthetic_classification),
    Experiment("Cover type", "multi:softmax", "merror", data_loader.get_cover_type),
    Experiment("YearPredictMSD", "reg:linear", "rmse", data_loader.get_year),
    Experiment("Higgs", "binary:logistic", "error", data_loader.get_higgs),
    Experiment("Bosch", "binary:logistic", "auc", data_loader.get_bosch),
    Experiment("Airline", "binary:logistic", "error", data_loader.get_airline),
]

# num_rows = None
num_rows = 10000
for exp in experiments:
    exp.run(num_rows)
