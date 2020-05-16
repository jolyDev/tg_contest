import algo_manager as algo
import common

training_data = algo.generate_test_data(clusters_num = 2, points_in_cluster = 25)

report = algo.analyze(data = training_data, draw_results = True)

report.sort()

common.printAlgosInfo(report)

# on this data we will check if algorithm puts items in correspond clusters
test_data = []
