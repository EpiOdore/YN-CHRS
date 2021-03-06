"""
Script python pour ouvrir les fichiers de traces de clavier

"""
import mean_clustering
import post_treatment
import pre_treatment
import CNN1D
import os
from os.path import isfile, join
from sklearn.cluster import AgglomerativeClustering, KMeans

import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import tensorflow as tf
import json


def read_int(f):
    ba = bytearray(4)
    f.readinto(ba)
    prm = np.frombuffer(ba, dtype=np.int32)
    return prm[0]


def read_double(f):
    ba = bytearray(8)
    f.readinto(ba)
    prm = np.frombuffer(ba, dtype=np.double)
    return prm[0]


def read_double_tab(f, n):
    ba = bytearray(8*n)
    nr = f.readinto(ba)
    if nr != len(ba):
        return []
    else:
        prm = np.frombuffer(ba, dtype=np.double)
        return prm


def get_pics_from_file(filename):
    # Lecture du fichier d'infos + pics detectes (post-processing KeyFinder)
    print("Ouverture du fichier de pics "+filename)
    f_pic = open(filename, "rb")
    info = dict()
    info["nb_pics"] = read_int(f_pic)
    print("Nb pics par trame: " + str(info["nb_pics"]))
    info["freq_sampling_khz"] = read_double(f_pic)
    print("Frequence d'echantillonnage: " +
          str(info["freq_sampling_khz"]) + " kHz")
    info["freq_trame_hz"] = read_double(f_pic)
    print("Frequence trame: " + str(info["freq_trame_hz"]) + " Hz")
    info["freq_pic_khz"] = read_double(f_pic)
    print("Frequence pic: " + str(info["freq_pic_khz"]) + " kHz")
    info["norm_fact"] = read_double(f_pic)
    print("Facteur de normalisation: " + str(info["norm_fact"]))
    tab_pics = []
    pics = read_double_tab(f_pic, info["nb_pics"])
    nb_trames = 1
    while len(pics) > 0:
        nb_trames = nb_trames+1
        tab_pics.append(pics)
        pics = read_double_tab(f_pic, info["nb_pics"])
    print("Nb trames: " + str(nb_trames))
    f_pic.close()
    return tab_pics, info


# return the data from the binary files
# IN: path: str, percent: float
# OUT: return dico{filenames(str): [trames, metadata]}
def get_all_bin(path):
    all_files = [f for f in os.listdir(path) if isfile(join(path, f))]

    print(all_files)

    dico_files_content = {}

    for file in all_files:
        dico_files_content[os.path.splitext(
            file)[0]] = get_pics_from_file(path + file)

    return dico_files_content


# concatenate trames from all the letters
# IN: path: str, percent: float
# OUT: list(float)
def get_letters_trames(path, percent):
    dico = get_all_bin(path)
    name = "A"
    trames = []
    while (name <= "Z"):
        trames += dico["pics_" +
                       name][0][0:int(len(dico["pics_" + name][0]) * percent)]
        name = chr(ord(name) + 1)
    return trames


# Run the clustering algorithm and return the label for each trame
# IN: listoflists(list(list(float)), number of cluster: int
# OUT: list(int)
def get_clusters(listoflists, number):
    listoflists = np.array(listoflists)
    clustering = AgglomerativeClustering(
        n_clusters=number).fit_predict(listoflists)
    return clustering


# Print in terminal perf of clustering
# IN: list_trames list(int), percent: float, dico_data:{name_bin_file(str): data from the bin file}, letter: bool
# OUT: None
def print_info_perf(list_trames, percent, dico_data, letter=True):
    if not letter:
        return

    name = "A"
    start = 0

    master_ratio = []
    clusters_detected = []

    for i in range(1, 27):
        sub_list = list_trames[start:start +
                               int(len(dico_data["pics_" + name][0]) * percent)]
        ratio = sub_list.count(
            max(sub_list, key=sub_list.count)) / len(sub_list)
        cluster = max(sub_list, key=sub_list.count)
        print("Letter " + name + " success percentage: ",
              ratio, " Cluster: ", cluster)
        name = chr(ord(name) + 1)
        start += len(sub_list)

        if cluster in clusters_detected:
            ratio = 1 - ratio
        else:
            clusters_detected.append(cluster)

        master_ratio.append(ratio)

    print("Master ratio: ", sum(master_ratio) / len(master_ratio))


# Run the clustering program and save the result in a csv
# IN: percent of the trames used: float
# OUT: None
def run_clustering(percent):
    file = open("statictrames-0_2.csv", 'w')
    write = csv.writer(file)

    trames = get_letters_trames("../data/", percent)
    start = time.perf_counter()
    clusters = list(get_clusters(trames, 26))
    print(clusters)
    end = time.perf_counter()

    print("running time: ", end - start)
    write.writerow(clusters)
    file.close()


# return list of int from the csv
# IN: name of the csv: str
# OUT: list(int)
def read_csv(name):
    with open(name, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    return [int(i) for i in data[0]]


def save_output(resultstring, filename):
    file = open(filename, "w")
    file.write(resultstring)
    file.close()


def run_CNN1D(network, dicoequivalences, inputlist):
    finalString = ''
    finalList = []
    for i in range((len(inputlist) - 3)):
        sample = np.array([inputlist[i], inputlist[i + 1],
                          inputlist[i + 2], inputlist[i + 3]])
        test = network.predict(np.array([sample]))
        output = test[0]
        maxval = max(output)
        maxpos = np.where(output == maxval)[0][0]
        foundcarac = dicoequivalences[maxpos]
        finalList.append(foundcarac)
        letter = foundcarac.split("_")
        if letter[1] != "NOKEY":
            finalString += letter[1]
        if i % 10 == 0:
            finalString += '\n'
        else:
            finalString += ' '
    return finalString, finalList


def run_on_all_char(dico_trames, network, dicoequivalences, name=None):
    if name != None:
        try:
            f = open(str(name), "r")
            print("Here")
            content = f.read()
            f.close()
            return json.loads(content)
        except Exception as e:
            print(e.args)

    keyList = dico_trames.keys()
    allWeightsDico = {}
    for key, value in dico_trames.items():
        weightList = []
        outputString, outputList = run_CNN1D(
            network, dicoequivalences, value[0])
        for key2 in keyList:
            keyCount = outputList.count(key2)
            if keyCount != 0:
                weightList.append([key2, keyCount])
        for i in range(len(weightList)):
            weightList[i][1] = weightList[i][1] / len(outputList)
        allWeightsDico[key] = weightList

    if name != None:
        f = open(name, "w")
        f.write(str(allWeightsDico).replace("\'", '"'))
        f.close()
        print(name + "file saved")

    return allWeightsDico


def get_proportions_in_split(split_list, dico_trames):
    keyList = dico_trames.values()
    result = []
    for list in split_list:
        list_proportions = []
        for key in keyList:
            keyCount = list.count(key)
            if keyCount != 0:
                list_proportions.append([key, keyCount / len(list)])
        result.append(list_proportions)
    return result


def get_model_list(nb_models, nb_pack, train_percent, dico_trames, new_train=False, save_train=False):
    if new_train:
        return [CNN1D.neural_network_1D(dico_trames, train_percent, i, nb_pack, save_train)[0] for i in range(nb_models)]
    else:
        return [tf.keras.models.load_model("./model_weight-" + str(i)) for i in range(nb_models)]


def use_models(models_list, dico_trames, nb_pack, input, train_percent):
    output = []
    for i in range(len(models_list)):
        (outputString, output_list) = run_CNN1D(models_list[i], CNN1D.trunc_dataset_1D(
            dico_trames, train_percent, nb_pack)[4], input)
        output.append((outputString, output_list))
    return output


def split_output_list(output_list):
    list_of_pools = []
    pool = []
    for i in range(len(output_list) - 7):
        if output_list[i] != "pics_NOKEY" and output_list[i + 1] != "pics_NOKEY" and output_list[i + 2] != "pics_NOKEY" and output_list[i + 3] != "pics_NOKEY" and output_list[i + 4] != "pics_NOKEY" and output_list[i + 5] != "pics_NOKEY" and output_list[i + 6] != "pics_NOKEY":
            pool.append(output_list[i + 2])
        else:
            if pool != []:
                list_of_pools.append(pool)
                pool = []
    return list_of_pools


def get_correspondance_cluster_file_dico(dico_trames):
    correspondance_dico = {}
    cluster_counter = float(0)

    for key, values in dico_trames.items():
        correspondance_dico[cluster_counter] = key
        cluster_counter += 1

    return correspondance_dico

def run_project(train_percent, nb_models, nb_pack, add_bruit=False, new_train=False, save_train = False):
    dico_trames = get_all_bin("../data/")
    if add_bruit:
        dico_trames = pre_treatment.addBruitGaussien(dico_trames)
    loginmdp = dico_trames.pop("pics_LOGINMDP")
    corresp_cluster_file_dico = get_correspondance_cluster_file_dico(
        dico_trames)
    list_models = get_model_list(nb_models, nb_pack, train_percent, dico_trames, new_train, save_train)
    print("Loading models")
    output = use_models(list_models, dico_trames, nb_pack, loginmdp[0], train_percent)
    print("Input treated")
    split_output = split_output_list(output[0][1])
    split_output_proportions = get_proportions_in_split(
        split_output, corresp_cluster_file_dico)
    print("Output splitted")
    frames_results_per_models = [run_on_all_char(
        dico_trames, list_models[i], corresp_cluster_file_dico, "model_stat-" + str(i)) for i in range(len(list_models))]
    print("Get models stats")

    packed_results_per_model = []
    for i in range(len(frames_results_per_models)):
        packed_results_per_model += [post_treatment.compare_model_test_n_result(
            frames_results_per_models[i], split_output_proportions, corresp_cluster_file_dico)]
        print("Model " + str(i) + " done")

    for result in packed_results_per_model:
        print(result)


if __name__ == "__main__":
    run_project(0.8, 1, 4, True, True)

