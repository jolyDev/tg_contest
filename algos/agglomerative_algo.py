# https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering

from sklearn.cluster import AgglomerativeClustering
import common

class AgglomerativeClustering_algo_wrapper:
    def __init__(self):
        self.wrapped = AgglomerativeClustering(linkage="complete", n_clusters=2)
        self.data = []
        self.indexes =[]

    def fit(self,data):
        self.wrapped.fit(data)
        self.data = data
        self.indexes = self.wrapped.labels_

    def predict(self,data):
        return self.wrapped.fit_predict(data)


def do(input_data, draw_plot = False) -> common.AlgoInfo:
    model = AgglomerativeClustering_algo_wrapper()
    model.fit(input_data)
    if draw_plot:
        common.draw(model.data, model.indexes)
    return common.AlgoInfo("AgglomerativeClustering", model.indexes)