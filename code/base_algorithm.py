from sklearn.ensemble import GradientBoostingRegressor
from node import Node
import numpy as np

class Predictor(object):
    def learn_predict(self, x_train, y_train, x_test):
        pass

    def get_name(self):
        pass


class PredictorFunction(Predictor):
    def __init__(self, root):
        self.root = root

    def learn_predict(self, x_train, y_train, x_test):
        y_test = []
        for i in range(x_test.shape[0]):
            y_test.append(self.root.count(np.asarray(x_test[i])))

        return np.asarray(y_test)

    def get_name(self):
        return str(self.root)


class PredictorGBR(Predictor):
    def __init__(self, n_estimators=100, random_state=1, verbose=1, min_samples_leaf=10, max_depth=3):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.verbose = verbose
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

    def learn_predict(self, x_train, y_train, x_test):
        gbr = GradientBoostingRegressor(n_estimators=self.n_estimators,
                                        random_state=self.random_state,
                                        verbose=self.verbose,
                                        min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth)
        gbr.fit(x_train, y_train)
        return gbr.predict(x_test)

    def get_name(self):
        return ' '.join(['GBR', str(self.n_estimators), str(self.max_depth)])