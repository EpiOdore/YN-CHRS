import numpy as np


# Algorithm which calculate the mean of all trames for a letter
# IN: dico_data: {filename: (trames, metadata)}
# OUT: list of means list(float), dico of LOGINMPD
def mean_clustering(dico_data):
    LOGMDP = dico_data["pics_LOGINMDP"][0]
    dico_data.pop("pics_LOGINMDP")

    mean_trames_list = []

    for key in dico_data:
        if key == "pics_LOGINMDP":
            continue
        trames = dico_data[key][0]
        means_coord = np.zeros(len(trames[0]))
        for trame in trames:
            for i_coord in range(len(trame)):
                means_coord[i_coord] += trame[i_coord]

        mean_func = lambda x: x / len(trames)
        means_coord = list(map(mean_func, means_coord))

        mean_trames_list.append(means_coord)

    return mean_trames_list, LOGMDP


# Find the mean_trame which has the best correspondance with the list of test trames
# IN: mean_trames_list: list(list(float)), test_trames: list(list(float))
# OUT: percentage: float, cluster: int
def test_mean_clustering(mean_trames_list, test_trames):
    res_trames = []

    for trames in test_trames:
        temp_mean_trames_list = list(map(sum, list(map(lambda x: list(map(abs, list(x - trames))), list(mean_trames_list)))))
        res_trames.append(temp_mean_trames_list.index(min(temp_mean_trames_list)))

    return res_trames.count(max(res_trames, key=res_trames.count)) / len(res_trames), max(res_trames, key=res_trames.count)


# show perf of mean clustering for each file
# IN: mean_trames_list: list(list(float)), dico_data: dico_binary_file
# OUT: None
def print_info_perf_mean(mean_trames_list, dico_data):

    for key in dico_data:
        res = test_mean_clustering(mean_trames_list, dico_data[key][0])
        print("File " + key + " success percentage: ", res[0], " Cluster: ", res[1])
