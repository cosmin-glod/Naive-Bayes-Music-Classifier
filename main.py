import numpy as np
import random
from GaussianBayes import gaussianNaiveBayes as gnb

def split_data():
    data = np.load("./pca.npy")
    data_labels = np.load("./labels.npy")

    ratio = 99 # 80:20 ratio training:test data
    dataset = np.zeros(data.shape[0], dtype=np.short)

    for i in range(1, 11):
        min_ = (i - 1) * 100 - (i // 7)
        max_ = min_ + 100 - (i % 6 == 0)
        train_choice = random.sample(range(min_, max_), ratio)
        dataset[train_choice] = 1

    train_idx = np.where(dataset == 1)
    train_data = data[train_idx]
    train_labels = data_labels[train_idx]

    test_idx = np.where(dataset == 0)
    test_data = data[test_idx]
    true_labels = data_labels[test_idx]

    return train_data, train_labels, test_data, true_labels, dataset

train_data, train_labels, test_data, true_labels, dataset = split_data()
predictions = gnb(train_data, train_labels, test_data)
# eva = evaluation(predictions, true_labels, name)
# predictions = np.argmax(predictions, axis = 0)
