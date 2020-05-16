# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

# general overview
# https://scikit-learn.org/stable/modules/clustering.html

import matplotlib.pyplot as plot

colors = ('#FF0000', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')

class AlgoInfo:
    def __init__(self, algoname, clusters):
        self.clusters = clusters
        self.name = algoname
        self.time_taken = 0

    def __lt__(self, other):
        return self.time_taken < other.time_taken

def printAlgosInfo(list):
    for item in list:
        print("{} [t: {} sec]".format(item.name, item.time_taken))

def draw(data, cluster_group):
    for i in range(len(data)):
        plot.scatter(data[i][0], data[i][1], 2, edgecolors=colors[int(cluster_group[i])])
    plot.show()
