import numpy as np


def get_key(to_find, dico):
    for key, value in dico.items():
        if value == to_find:
            return key


def output_framed_stat_to_coord(output_framed_stat, correspondance_dico):
    coord_list = []
    # print(output_framed_stat)
    # print(correspondance_dico)
    for block in output_framed_stat:
        frame_coord = np.zeros(len(correspondance_dico))

        for tupple in block:
             frame_coord[int(get_key(tupple[0], correspondance_dico))] = tupple[1]

        coord_list += [frame_coord]

    return coord_list


def model_stat_coord(model_stat, correspondance_dico):

    key_coord_dico = [[] for _ in range(len(correspondance_dico))]
    for key, stats in model_stat.items():
        frame_coord = np.zeros(len(correspondance_dico))
        # print(key_results[1])
        # print(model_stat)
        # print(key_results)

        for tupple in stats:
            frame_coord[int(get_key(tupple[0], correspondance_dico))] = tupple[1]

        key_coord_dico[int(get_key(key, correspondance_dico))] = frame_coord

    return key_coord_dico


def dico_to_list(point_a, point_b):
    list_a = []
    list_b = []
    correspondance_dico = {}
    for key in point_a.items():
        correspondance_dico[len(list_b)] = key
        list_a += point_a[key]
        list_b += point_b[key]

    return list_a, list_b, correspondance_dico


def compare_model_test_n_result(model_stat, output_framed_stat, correspondance_dico):
    frame_coord_list = output_framed_stat_to_coord(output_framed_stat, correspondance_dico)
    model_coord_results = model_stat_coord(model_stat, correspondance_dico)

    results_close_to_coord_list = []
    print("model and frame casted")

    for frame_coord in frame_coord_list:
        list_dist = np.zeros(len(correspondance_dico))
        for i in range(len(model_coord_results)):
            list_dist[i] = np.linalg.norm(np.array(frame_coord)-np.array(model_coord_results[i]))

        minus_key = list(list_dist).index(min(list_dist))
        print()
        for key, value in correspondance_dico.items():
            if int(key) == minus_key:
                results_close_to_coord_list.append(value)
                break

    print(results_close_to_coord_list)
    return results_close_to_coord_list


def get_maximum(_list):
    nb_occ = dict((x, _list.count(x)) for x in set(_list))
    occ_max = max(nb_occ.values())
    maxi = []

    for key, value in nb_occ.items():
        if value == occ_max:
            maxi += key

    return maxi


def compare_results_per_model(results_per_model):

    results = []
    if len(results_per_model) == 0:
        return results
    for i in range(len(results_per_model[0])):
        pack_result = []
        for model_result in results_per_model:
            pack_result += model_result[i]

        results += get_maximum(pack_result)

    return results
