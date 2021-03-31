import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


def gaussian_kernel_trames(dico_data, key, percent):
    trames_train = dico_data[key][0][0: int(len(dico_data[key][0]) * percent)]
    # params = {'bandwidth': np.logspace(-1, 1, 20)}
    # grid = GridSearchCV(KernelDensity(), params)
    # grid.fit(trames)
    # print(key + " best bandwith: ", grid.best_estimator_.bandwidth, " best kernel: ", grid.best_estimator_.kernel)
    return KernelDensity(kernel='gaussian', bandwidth=0.1).fit(trames_train)


def mean_trames(dico_data, key):
    trames = dico_data[key][0]
    means_coord = np.zeros(len(trames[0]))
    for trame in trames:
        for i_coord in range(len(trame)):
            means_coord[i_coord] += trame[i_coord]

    mean_func = lambda x: x / len(trames)

    return list(map(mean_func, means_coord))


# Algorithm which calculate the mean of all trames for a letter
# IN: dico_data: {filename: (trames, metadata)}, mean:bool
# OUT: list of means list(float), dico of LOGINMPD
def mean_clustering(dico_data, percent, mean=True):
    LOGMDP = dico_data["pics_LOGINMDP"][0]
    dico_data.pop("pics_LOGINMDP")

    mean_trames_list = {}

    for key in dico_data:
        if key == "pics_LOGINMDP":
            continue
        if mean:
            means_coord = mean_trames(dico_data, key)
        else:
            means_coord = gaussian_kernel_trames(dico_data, key, percent)

        mean_trames_list[key] = means_coord

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


def test_gaussian_kernel_clustering(gaussian_dico, test_trames, filename):
    # train = test_trames[0: int(len(test_trames) * 0.8)]
    test = test_trames[int(len(test_trames) * 0.8):]
    score = 0
    for trame in test:
        score_dico = {}
        for key in gaussian_dico:
            score_dico[key] = np.exp(gaussian_dico[key].score([trame]))

        detected_value = max(score_dico, key=score_dico.get)
        score += int(detected_value == filename)
        # print(detected_value)
        # test_dico[key] = gaussian_dico[key].score_samples(test_trames)
    return score / len(test)

    minus = min(test_dico, key=test_dico.get)

    for key, value in test_dico.items():
        if value == key:
            return minus, key


def print_info_perfo_gauss(gaussian_dico, trames_dico):

    success = 0

    trames_dico.pop("pics_NOKEY")
    gaussian_dico.pop   ("pics_NOKEY")

    for filename, data in trames_dico.items():
        ratio = test_gaussian_kernel_clustering(gaussian_dico, data[0], filename)
        print("File tested: " + filename + " ratio: ", ratio)
        success += ratio

    print("Master ratio: ", success / len(trames_dico))


# show perf of mean clustering for each file
# IN: mean_trames_list: list(list(float)), dico_data: dico_binary_file
# OUT: None
def print_info_perf_mean(mean_trames_list, dico_data):

    for key in dico_data:
        res = test_mean_clustering(mean_trames_list, dico_data[key][0])
        print("File " + key + " success percentage: ", res[0], " Cluster: ", res[1])
