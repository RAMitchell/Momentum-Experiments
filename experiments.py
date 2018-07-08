import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

import ml_dataset_loader.datasets as data_loader

test_size = 0.2
train_size = 0.6
validation_size = 0.2
random_seed = 7
num_rounds = 20000
early_stopping_rounds = 50

common_param = {'eta': 0.05, 'silent': 1, 'tree_method': 'gpu_hist', 'subsample': 0.9,
                'colsample_bytree': 0.7}
optimizer_param = [
    {'name': 'Baseline', 'optimizer': 'default_optimizer'},
    {'name': 'Momentum', 'optimizer': 'momentum_optimizer', 'momentum': 0.3},
    {'name': 'Nesterov', 'optimizer': 'nesterov_optimizer', 'momentum': 0.3},
]


class Experiment:
    def __init__(self, name, objective, metric, load_data):
        self.name = name
        self.objective = objective
        self.metric = metric
        self.load_data = load_data

    def run(self, num_rows=None):
        # Create train/test/validation sets
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

        data = {}
        plt.clf()
        for opt_param in optimizer_param:
            param = common_param.copy()
            param.update(opt_param)
            param["objective"] = self.objective
            param["eval_metric"] = self.metric
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
            data[param["name"]] = res['test'][self.metric][:bst.best_iteration + 1]
            # Plot series
            plt.plot(data[param["name"]], marker='x', markevery=[bst.best_iteration],
                     label=param["name"])

        # Plot
        plt.xlabel('iterations')
        plt.ylabel(self.metric)
        plt.legend()
        plt.title(self.name)
        snake_name = re.sub(' ', '_', self.name).lower()
        plt.savefig('figures/' + snake_name + '.pgf', bbox_inches='tight')
        plt.savefig('figures/' + snake_name + '.png', bbox_inches='tight')

        # Zoom
        plt.xlim(1e15, 0)
        for name, series in data.items():
            xmin, xmax = plt.xlim()
            plt.xlim(min(xmin, len(series) - 1), max(xmax, len(series) - 1))

        plt.ylim(1e15, 0)
        for name, series in data.items():
            ymin, ymax = plt.ylim()
            plt.ylim(min(ymin, series[-1]), max(ymax, series[-1]))

        # Add padding
        padding = 0.35
        xmin, xmax = plt.xlim()
        xpad = abs(xmin - xmax) * padding
        plt.xlim(xmin - xpad, xmax + xpad)
        ymin, ymax = plt.ylim()
        ypad = abs(ymin - ymax) * padding
        plt.ylim(ymin - ypad, ymax + ypad)

        plt.savefig('figures/' + snake_name + '_zoomed.pgf', bbox_inches='tight')
        plt.savefig('figures/' + snake_name + '_zoomed.png', bbox_inches='tight')


num_rows = 500000
experiments = [
    Experiment("Synthetic regression", "reg:linear", "rmse", data_loader.get_synthetic_regression),
    Experiment("Cover type", "multi:softmax", "merror", data_loader.get_cover_type),
    Experiment("YearPredictMSD", "reg:linear", "rmse", data_loader.get_year),
    Experiment("Higgs", "binary:logistic", "error", data_loader.get_higgs),
    Experiment("Bosch", "binary:logistic", "auc", data_loader.get_bosch),
    Experiment("Airline", "binary:logistic", "error", data_loader.get_airline),
]

matplotlib.use('pgf')
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],  # use latex default serif font
})

for exp in experiments:
    exp.run(num_rows)
