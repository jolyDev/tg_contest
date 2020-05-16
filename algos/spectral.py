# Warning this algo only for 2 dimentional arrays !!!
# https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering

from sklearn.cluster import SpectralClustering
import common

class SpectralClustering_algo_wrapper:
    def __init__(self):
        self.wrapped = SpectralClustering(2, affinity='precomputed', n_init=100, assign_labels='discretize')
        self.data = []
        self.indexes =[]

    def fit(self,data):
        self.wrapped.fit(data)
        self.data = data
        self.indexes = self.wrapped.labels_

    def predict(self,data):
        return self.wrapped.fit_predict(data)


def do(input_data, draw_plot = False) -> common.AlgoInfo:
    model = SpectralClustering_algo_wrapper()
    model.fit(input_data)
    if draw_plot:
        common.draw(model.data, model.indexes)
    return common.AlgoInfo("SpectralClustering", model.indexes)