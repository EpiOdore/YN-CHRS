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


def neural_network(dico_trames, percent):
    print(len(dico_trames))
    truncated_dataset = trunc_dataset(dico_trames, percent)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(17, input_dim=17, activation='relu'),
        tf.keras.layers.Dense(17 * 2, activation='relu'),
        tf.keras.layers.Dense(int(17 * 0.6 + 42), activation='relu'),
        # tf.keras.layers.Dense(int(17 * 0.6 + 42), activation='relu'),
        tf.keras.layers.Dense(42, activation='relu'),
        tf.keras.layers.Dense(42, activation='softmax')
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

    model.fit(truncated_dataset[0], truncated_dataset[1], epochs=80)

    _, train_accuracy = model.evaluate(truncated_dataset[0], truncated_dataset[1])
    _, test_accuracy = model.evaluate(truncated_dataset[2], truncated_dataset[3])

    print("Accuracy on train trames: ", train_accuracy)
    print("Accuracy on test trames: ", test_accuracy)

