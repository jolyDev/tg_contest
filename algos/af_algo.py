# https://scikit-learn.org/stable/auto_examples/cluster/plot_affinity_propagation.html#sphx-glr-auto-examples-cluster-plot-affinity-propagation-py

from sklearn.cluster import AffinityPropagation
import common

class AffinityPropagation_algo_wrapper:
    def __init__(self):
        self.wrapped = AffinityPropagation()
        self.data = []
        self.indexes =[]

    def fit(self,data):
        self.wrapped.fit(data)
        self.data = data
        self.indexes = self.wrapped.labels_

    def predict(self,data):
        return self.wrapped.fit_predict(data)


def do(input_data, draw_plot = False) -> common.AlgoInfo:
    model = AffinityPropagation_algo_wrapper()
    model.fit(input_data)
    if draw_plot:
        common.draw(model.data, model.indexes)
    return common.AlgoInfo("AffinityPropagation", model.indexes)