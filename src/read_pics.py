"""
Script python pour ouvrir les fichiers de traces de clavier

"""
import mean_clustering
# import neural_network
import CNN1D
import os
from os.path import isfile, join
from sklearn.cluster import AgglomerativeClustering, KMeans

import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import tensorflow as tf


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
    print("Frequence d'echantillonnage: " + str(info["freq_sampling_khz"]) + " kHz")
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
        dico_files_content[os.path.splitext(file)[0]] = get_pics_from_file(path + file)

    return dico_files_content


# concatenate trames from all the letters
# IN: path: str, percent: float
# OUT: list(float)
def get_letters_trames(path, percent):
    dico = get_all_bin(path)
    name = "A"
    trames = []
    while (name <= "Z"):
        trames += dico["pics_" + name][0][0:int(len(dico["pics_" + name][0]) * percent)]
        name = chr(ord(name) + 1)
    return trames


# Run the clustering algorithm and return the label for each trame
# IN: listoflists(list(list(float)), number of cluster: int
# OUT: list(int)
def get_clusters(listoflists, number):
    listoflists = np.array(listoflists)
    clustering = AgglomerativeClustering(n_clusters=number).fit_predict(listoflists)
    # clustering = KMeans(n_clusters=number).fit_predict(listoflists)
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
        sub_list = list_trames[start:start + int(len(dico_data["pics_" + name][0]) * percent)]
        ratio = sub_list.count(max(sub_list, key=sub_list.count)) / len(sub_list)
        cluster = max(sub_list, key=sub_list.count)
        print("Letter " + name + " success percentage: ", ratio, " Cluster: ", cluster)
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
    for i in range(len(inputlist) - 3):
        sample = np.array([inputlist[i], inputlist[i + 1], inputlist[i + 2], inputlist[i + 3]])
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


def run_on_all_char(dico_trames, network, dicoequivalences):
    keyList = dico_trames.keys()
    allWeightsDico = {}
    for key, value in dico_trames.items() :
        weightList = []
        outputString, outputList = run_CNN1D(network, dicoequivalences, value[0])
        for key2 in keyList:
            keyCount = outputList.count(key2)
            if keyCount != 0:
                weightList.append([key2, keyCount])
        for i in range(len(weightList)):
            weightList[i][1] = weightList[i][1] / len(outputList)
        allWeightsDico[key] = weightList
    return allWeightsDico


def get_model_list(nb_models, nb_pack, train_percent, new_train=False):
    if new_train:
        return [CNN1D.neural_network_1D(dico_trames, train_percent, i, nb_pack)[0] for i in range(nb_models)]
    else:
        return [tf.keras.models.load_model("./model_weight-" + str(i)) for i in range(nb_models)]


def feed_models(models_list, dico_trames, nb_pack, input):
    for i in range(len(models_list)):
        (outputString, output_list) = run_CNN1D(list_models[i], CNN1D.trunc_dataset_1D(dico_trames, train_percent, nb_pack)[4], input)
        print(outputString)


if __name__ == "__main__":
    train_percent = 0.8
    mean = False
    new_train = False
    nb_models = 4
    nb_pack = 4
    # run_clustering(percent)
    dico_trames = get_all_bin("../data/")
    all_trames = read_csv('statictrames-0_2.csv')
    loginmdp = dico_trames.pop("pics_LOGINMDP")
    # print_info_perf(all_trames, percent, dico_trames)
    # analysis_list, LOGMDP = mean_clustering.mean_clustering(dico_trames, percent, mean)
    # outputString = run_CNN1D(network, dicoequivalences, loginmdp[0])
    list_models = get_model_list(nb_models, nb_pack, train_percent, new_train)
    feed_models(list_models, dico_trames, nb_pack, loginmdp[0])

        # save_output(outputString, "outputV3-" + str(i) +".txt")

    # neural_network.neural_network(dico_trames, percent)

    # if mean:
    #     mean_clustering.test_mean_clustering(analysis_list, dico_trames["pics_M"][0])
    #     mean_clustering.print_info_perf_mean(analysis_list, dico_trames)
    # else:
    #     print(dico_trames["pics_M"][0])
        # mean_clustering.print_info_perfo_gauss(analysis_list, dico_trames)
        # mean_clustering.test_gaussian_kernel_clustering(analysis_list, LOGMDP, "pics_G")
        # mean_clustering.test_gaussian_kernel_clustering(analysis_list, dico_trames["pics_G"][0], "pics_G")



    """pics_nokey, info = get_pics_from_file("../data/pics_NOKEY.bin")
    pics_pad0, info = get_pics_from_file("../data/pics_0.bin")

    ######### Pics ############
    # PAD-0 0
    plt.figure(1)
    plt.subplot(311)
    plt.plot(range(1,info["nb_pics"]+1), pics_nokey[0], 'ko')
    plt.xlabel('numéro de pic')
    plt.ylabel('valeur du pic')
    plt.title('PAD-0 0')
    plt.ylim(0, 1.5)
    plt.grid(b=True, which='both')
    # PAD-0 1
    plt.subplot(312)
    plt.plot(range(1,info["nb_pics"]+1), pics_nokey[1], 'ko')
    plt.xlabel('numéro de pic')
    plt.ylabel('valeur du pic')
    plt.title('PAD-0 1')
    plt.ylim(0, 1.5)
    plt.grid(b=True, which='both')
    # PAD-0 2
    plt.subplot(313)
    plt.plot(range(1,info["nb_pics"]+1), pics_nokey[2], 'ko')
    plt.xlabel('numéro de pic')
    plt.ylabel('valeur du pic')
    plt.title('PAD-0 2')
    plt.ylim(0, 1.5)
    plt.grid(b=True, which='both')
    #
    plt.show()"""
