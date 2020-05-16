# https://github.com/scikit-learn/scikit-learn/tree/483cd3eaa3c636a57ebb0dc4765531183b274df0/sklearn/cluster

from sklearn.cluster import KMeans
import common

class K_Means_wrapper:
    def __init__(self):
        self.wrapped = KMeans(n_clusters=2)
        self.data = []
        self.indexes =[]

    def fit(self,data):
        self.wrapped.fit(data)
        self.indexes = self.wrapped.labels_
        self.data = data

    def predict(self,data):
        return self.wrapped.predict(data)


def do(input_data, draw_plot = False) -> common.AlgoInfo:
    model = K_Means_wrapper()
    model.fit(input_data)
    if draw_plot:
        common.draw(model.data, model.indexes)
    return common.AlgoInfo("K_mean", model.indexes);