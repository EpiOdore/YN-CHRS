import numpy as np


def get_key(to_find, dico):
    for key, value in dico.items():
        if value == to_find:
            return key


def output_framed_stat_to_coord(output_framed_stat, correspondance_dico):
    coord_list = []
    print(output_framed_stat)
    print(correspondance_dico)
    for key, prob in output_framed_stat.items():
        frame_coord = np.zeros(len(correspondance_dico))

        for tupple in prob:
             frame_coord[int(get_key(tupple[0], correspondance_dico))] = tupple[1]

        coord_list += [frame_coord]

    return coord_list


def model_stat_coord(model_stat, correspondance_dico):

    key_coord_dico = np.zeros(len(correspondance_dico))
    for key_results in model_stat:
        frame_coord = np.zeros(len(correspondance_dico))

        for tupple in key_results[1]:
            frame_coord[int(get_key(key_results[0], correspondance_dico))] = tupple[1]

        key_coord_dico[int(get_key(key_results[0], correspondance_dico))] = frame_coord

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

    for frame_coord in frame_coord_list:
        list_dist = np.zeros(len(correspondance_dico))
        for i in range(len(model_coord_results)):
            list_dist[i] = np.linalg.norm(np.array(frame_coord)-np.array(model_coord_results[i]))

        minus_key = list(list_dist).index(min(list_dist))
        for key, value in correspondance_dico.items():
            if value == minus_key:
                results_close_to_coord_list += key
                break

    print(results_close_to_coord_list)