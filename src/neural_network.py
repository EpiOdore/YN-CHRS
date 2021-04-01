import tensorflow as tf
import numpy as np


def trunc_dataset(dico_trames, percent):
    total_length_train = 0
    total_length_test = 0
    for key, values in dico_trames.items():
        total_length_train += int(len(values[0]) * percent)
        total_length_test += int(len(values[0]) * (1 - percent))

    train_trames = []
    train_results = []
    test_trames = []
    test_results = []

    dico_corresp_cluster_file = {}
    cluster = float(0)
    for key, values in dico_trames.items():
        if cluster == float(10):
            break
        trames = values[0]
        train_trames += trames[0: int(len(trames) * percent)]
        test_trames += trames[int(len(trames) * (1 - percent)):]

        train_results += [cluster] * len(trames[0: int(len(trames) * percent)])
        test_results += [cluster] * len(trames[int(len(trames) * (1 - percent)):])

        dico_corresp_cluster_file[cluster] = key
        cluster += 1

    return np.array(train_trames), np.array(train_results), np.array(test_trames), np.array(test_results), dico_corresp_cluster_file


def concatenate_trames(trames):
    end_selected_trames = len(trames) - (len(trames) % 2)

    res_trames = []

    for i in range(0, end_selected_trames, 2):
        trames += [trames[i] + trames[i + 1]]

    return np.array(res_trames)


def cast_data(trames):
    for trame in trames:
        for j in range(len(trame)):
            trame[j] = np.array(trame[j])

    return trames


def packed_trames(trames, nb_pack):
    # trames = concatenate_trames(trames)
    # trames = cast_data(trames)
    end_selected_trames = len(trames) - (len(trames) % nb_pack)

    packs = []
    start = 0
    for i in range(0, end_selected_trames, nb_pack):
        if i == 0:
            continue
        packs += [np.array(trames[start:i])]
        start = i

    return np.array(packs)


def trunc_packed_dataset(dico_trames, percent, nb_pack):
    total_length_train = 0
    total_length_test = 0
    for key, values in dico_trames.items():
        total_length_train += int(len(values[0]) * percent)
        total_length_test += int(len(values[0]) * (1 - percent))

    train_trames = []
    train_results = []
    test_trames = []
    test_results = []

    dico_corresp_cluster_file = {}
    cluster = float(0)
    for key, values in dico_trames.items():
        if cluster == float(10):
            break
        trames = values[0]
        packs_train = packed_trames(trames[0: int(len(trames) * percent)], nb_pack)
        packs_test = packed_trames(trames[int(len(trames) * (1 - percent)):], nb_pack)
        train_trames += [packs_train]
        test_trames += [packs_test]
        # print(train_trames)
        # print(test_trames)
        # return

        train_results += [cluster] * len(packs_train)
        test_results += [cluster] * len(packs_test)

        dico_corresp_cluster_file[cluster] = key
        cluster += 1

    train_trames = [np.array(i) for i in train_trames]
    test_trames = [np.array(i) for i in test_trames]

    return np.array(train_trames), np.array(train_results), np.array(test_trames), np.array(test_results), dico_corresp_cluster_file


def neural_network(dico_trames, percent):
    # print(len(dico_trames))
    truncated_dataset = trunc_dataset(dico_trames, percent)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(17, input_dim=17, activation='relu'),
        # tf.keras.layers.Dense(17 * 2, activation='relu'),
        # tf.keras.layers.Dense(int(17 * 0.6 + 42), activation='relu'),
        tf.keras.layers.Dense(int(17 * 0.6 + 42), activation='relu'),
        # tf.keras.layers.Dense(42, activation='relu'),
        tf.keras.layers.Dense(42, activation='softmax')
    ])

    print(truncated_dataset[0][0])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

    model.fit(truncated_dataset[0], truncated_dataset[1], epochs=100)

    _, train_accuracy = model.evaluate(truncated_dataset[0], truncated_dataset[1])
    _, test_accuracy = model.evaluate(truncated_dataset[2], truncated_dataset[3])

    print("Accuracy on train trames: ", train_accuracy)
    print("Accuracy on test trames: ", test_accuracy)


def convolute_neural_network(dico_trames, percent):
    # print(len(dico_trames))
    truncated_dataset = trunc_packed_dataset(dico_trames, percent, 17)

    print(truncated_dataset[0])

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(17, (17, 1), activation='relu', input_shape=(17, 17, 1))])
        # tf.keras.layers.MaxPooling2D((1, 1)),
        # tf.keras.layers.Conv2D(17 * 2, (17, 1), activation='relu'),
        # tf.keras.layers.MaxPooling2D((1, 1)),
        # tf.keras.layers.Conv2D(17 * 2, (17, 1), activation='relu'),
        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(17 * 2, activation='relu'),
        # tf.keras.layers.Dense(42, activation='softmax')
    # ])

    print("HERE")

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

    model.fit(truncated_dataset[0], truncated_dataset[1], epochs=100)

    _, train_accuracy = model.evaluate(truncated_dataset[0], truncated_dataset[1])
    _, test_accuracy = model.evaluate(truncated_dataset[2], truncated_dataset[3])

    print("Accuracy on train trames: ", train_accuracy)
    print("Accuracy on test trames: ", test_accuracy)

