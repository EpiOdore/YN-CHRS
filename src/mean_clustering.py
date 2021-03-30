import numpy as np


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


def test_mean_clustering(mean_trames_list, test_trames):
    res_trames = []

    for trames in test_trames:
        temp_mean_trames_list = list(map(sum, list(map(lambda x: list(map(abs, list(x - trames))), list(mean_trames_list)))))
        res_trames.append(temp_mean_trames_list.index(min(temp_mean_trames_list)))

    return res_trames.count(max(res_trames, key=res_trames.count)) / len(res_trames), max(res_trames, key=res_trames.count)


def print_info_perf_mean(mean_trames_list, dico_data, letter=True):
    if not letter:
        return

    for key in dico_data:
        res = test_mean_clustering(mean_trames_list, dico_data[key][0])
        print("File " + key + " success percentage: ", res[0], " Cluster: ", res[1])
