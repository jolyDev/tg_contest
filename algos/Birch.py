# https://scikit-learn.org/stable/modules/clustering.html#birch

from sklearn.cluster import Birch
import common

class Birch_algo_wrapper:
    def __init__(self):
        self.wrapped = Birch(n_clusters = 2)
        self.data = []
        self.indexes =[]

    def fit(self,data):
        self.wrapped.fit(data)
        self.data = data
        self.indexes = self.wrapped.labels_

    def predict(self,data):
        return self.wrapped.predict(data)

model = Birch_algo_wrapper()

def do(input_data, draw_plot = False) -> common.AlgoInfo:
    model.fit(input_data)
    if draw_plot:
        common.draw(model.data, model.indexes)
    return common.AlgoInfo("Birch", model.indexes)


def predict(el) -> []:
    return model.predict(el)