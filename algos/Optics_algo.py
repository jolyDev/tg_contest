# https://scikit-learn.org/stable/modules/clustering.html#optics

from sklearn.cluster import OPTICS, cluster_optics_dbscan
import common

class OPTICS_algo_wrapper:
    def __init__(self):
        self.wrapped = OPTICS(min_samples=5, xi=.05, min_cluster_size=.05)
        self.data = []
        self.indexes =[]

    def fit(self,data):
        self.wrapped.fit(data)
        self.data = data
        self.indexes = self.wrapped.labels_

    def predict(self,data):
        return self.wrapped.fit_predict(data)


def do(input_data, draw_plot = False) -> common.AlgoInfo:
    model = OPTICS_algo_wrapper()
    model.fit(input_data)
    if draw_plot:
        common.draw(model.data, model.indexes)
    return common.AlgoInfo("OPTICS", model.indexes)