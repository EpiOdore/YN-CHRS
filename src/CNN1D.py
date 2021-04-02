from neural_network import trunc_dataset
import tensorflow as tf
import numpy as np


def pack_trames(trames, nb_pack):
    end_selected_trames = len(trames) - (len(trames) % nb_pack)

    packs = []
    start = 0
    for i in range(0, end_selected_trames, nb_pack):
        if i == 0:
            continue
        pack = trames[start:i]
        # for trame in pack:
        #     for j in range(len(trame)):
        #         trame[j] = np.array(trame[j])
        packs.append(np.array(pack))
        start = i
    # packs = np.array(packs)
    # print("test")
    return packs


def trunc_dataset_1D(dico_trames, percent, nb_pack):
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
        trames = values[0]
        train_pack = pack_trames(trames[0: int(len(trames) * percent)], nb_pack)
        test_pack = pack_trames(trames[int(len(trames) * (1 - percent)):], nb_pack)
        train_trames += train_pack
        test_trames += test_pack

        train_results += [cluster] * len(train_pack)
        test_results += [cluster] * len(test_pack)

        dico_corresp_cluster_file[cluster] = key
        cluster += 1

    return np.array(train_trames), np.array(train_results), np.array(test_trames), np.array(test_results), dico_corresp_cluster_file


def neural_network_1D(dico_trames, percent):
    # print(len(dico_trames))
    nb_pack = 4
    truncated_dataset = trunc_dataset_1D(dico_trames, percent, nb_pack)

    print(truncated_dataset[0].shape)

    model = tf.keras.Sequential()
    # model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
    model.add(tf.keras.layers.Conv1D(50, 2, activation='relu', input_shape=(nb_pack, 17)))
    model.add(tf.keras.layers.Conv1D(50, 2, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(2))
    model.add(tf.keras.layers.Conv1D(80, 1, activation='relu'))
    model.add(tf.keras.layers.Conv1D(80, 1, activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(42, activation='relu')) #-> Peut etre superflu
    model.add(tf.keras.layers.Dense(42, activation='softmax'))
    print(model.summary())

    # print(truncated_dataset[0][0])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

    model.fit(truncated_dataset[0], truncated_dataset[1], epochs=100, validation_data=(truncated_dataset[2], truncated_dataset[3]))

    _, train_accuracy = model.evaluate(truncated_dataset[0], truncated_dataset[1])
    _, test_accuracy = model.evaluate(truncated_dataset[2], truncated_dataset[3])

    print("Accuracy on train trames: ", train_accuracy)
    print("Accuracy on test trames: ", test_accuracy)
    return model, truncated_dataset[4]
