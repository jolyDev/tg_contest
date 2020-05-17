# https://scikit-learn.org/stable/modules/clustering.html#mean-shift

from sklearn.cluster import MeanShift, estimate_bandwidth
import common

class mean_shift_algo_wrapper:
    def __init__(self):
        self.wrapped = []
        self.data = []
        self.indexes =[]

    def fit(self,data):
        bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500)
        self.wrapped = MeanShift()
        self.wrapped.fit(data)
        self.data = data
        self.indexes = self.wrapped.labels_

    def predict(self,data):
        return self.wrapped.fit_predict(data)

model = mean_shift_algo_wrapper()

def do(input_data, draw_plot = False) -> common.AlgoInfo:
    model.fit(input_data)
    if draw_plot:
        common.draw(model.data, model.indexes)
    return common.AlgoInfo("Mean Shift", model.indexes)

def predict(el) -> []:
    return model.predict(el)