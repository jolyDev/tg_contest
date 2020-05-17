# https://scikit-learn.org/stable/modules/mixture.html#mixture

from sklearn import mixture
import common

class Gaussian_Mixture_algo_wrapper:
    def __init__(self):
        self.wrapped = mixture.GaussianMixture(n_components=2, covariance_type="tied")
        self.data = []
        self.indexes =[]

    def fit(self,data):
        self.wrapped.fit(data)
        self.data = data
        self.indexes = self.wrapped.labels_

    def predict(self,data):
        return self.wrapped.fit_predict(data)

model = Gaussian_Mixture_algo_wrapper()

def do(input_data, draw_plot = False) -> common.AlgoInfo:
    model.fit(input_data)
    if draw_plot:
        common.draw(model.data, model.indexes)
    return common.AlgoInfo("Gaussian Mixture", model.indexes)


def predict(el) -> []:
    return model.predict(el)