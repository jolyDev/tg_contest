# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN

from sklearn.cluster import DBSCAN
import common

class DBSCAN_algo_wrapper:
    def __init__(self):
        self.wrapped = DBSCAN(eps=3, min_samples=2)
        self.data = []
        self.indexes =[]

    def fit(self,data):
        self.wrapped.fit(data)
        self.data = data
        self.indexes = self.wrapped.labels_

    def predict(self,data):
        return self.wrapped.fit_predict(data)

model = DBSCAN_algo_wrapper()

def do(input_data, draw_plot = False) -> common.AlgoInfo:
    model.fit(input_data)
    if draw_plot:
        common.draw(model.data, model.indexes)
    return common.AlgoInfo("DBSCAN", model.indexes)


def predict(el) -> []:
    return model.predict(el)
