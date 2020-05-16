# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

from algos import\
    DBSCAN_algo as dbscan, \
    k_mean_algo as k_mean, \
    af_algo as af, \
    agglomerative_algo as ag, \
    Optics_algo as optics, \
    Birch as birch, \
    gaussian_mixture as gaussian, \
    mean_shift_algo as mean_shift, \
    spectral as spectral

import random
import common
import numpy as np
from timeit import default_timer as timer

INPUT_DATA = np.array([[1, 2],
                       [1.5, 1.8],
                       [5, 8],
                       [8, 8],
                       [1, 0.6],
                       [9, 11]])

algo_list = [af.do, birch.do, mean_shift.do, optics.do, ag.do, af.do, dbscan.do, k_mean.do]
algo_results_list = 0  # unnececaryly


def analyze(data, draw_results) -> []:
    report = []
    for i in range(len(algo_list)):
        start = timer()
        info = algo_list[i](data, draw_results)
        end = timer()
        info.time_taken = end - start
        report.append(info)
    return report

def generate_test_data(clusters_num, points_in_cluster) -> []:
    data = [[]]
    for cluster in range(clusters_num):
        x0 = random.random()
        y0 = random.random()
        for i in range(points_in_cluster):
            x = random.random() ** 2
            y = random.random() ** 2
            data.append([(x - x0) ** 2, (y - y0) ** 2])
    return np.array(data[1:])


