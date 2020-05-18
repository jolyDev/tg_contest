import algo_manager as algo
import common

training_data = algo.generate_test_data(clusters_num = 2, points_in_cluster = 25)

training_data = common.basic_test_n_1_two_claster_array_of_2d_points

report = algo.analyze(data = training_data, draw_results = False)


# on this data we will check if algorithm puts items in correspond clusters
compare_results = common.basic_test_n_1_diff
succes_rate = algo.compare_set_with_ideal_element( training_data, compare_results)

for i in range(len(succes_rate)):
    report[i].rate = succes_rate[i]

common.sortByTime(report)
common.sortByRate(report)

common.printAlgosInfo(report)