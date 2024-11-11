import numpy as np
import random
from featureExtraction import custom_compute_features as ccf
from featureReduction import PCA as pca
from GaussianBayes import gaussianNaiveBayes as gnb

def split_data(data_ratio):

    pca()

    data = np.load("./pca.npy")
    data_labels = np.load("./labels.npy")

    ratio = data_ratio
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

def calculate_metrics(metric_true_labels, metric_predictions, num_classes):
    precisions = []
    recalls = []
    f_scores = []

    for class_id in range(num_classes):
        # True Positives
        tp = np.sum((metric_predictions == class_id) & (metric_true_labels == class_id))

        # False Positives
        fp = np.sum((metric_predictions == class_id) & (metric_true_labels != class_id))

        # False Negatives
        fn = np.sum((metric_predictions != class_id) & (metric_true_labels == class_id))


        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precisions.append(precision)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recalls.append(recall)

        f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f_scores.append(f_score)

    return precisions, recalls, f_scores

def computing_statistics(data_true_labels, data_predictions):

    data_genres = {"blues": 0, "classical": 1, "country": 2,
                   "disco": 3, "hiphop": 4, "jazz": 5, "metal": 6,
                   "pop": 7, "reggae": 8, "rock": 9}

    new_true_labels = np.array([data_genres[genre] for genre in data_true_labels])
    new_predictions = np.argmax(data_predictions, axis=0)

    precisions, recalls, f_scores = calculate_metrics(new_true_labels, new_predictions, len(data_genres))
    accuracy = np.sum(new_predictions == new_true_labels) / len(new_true_labels)

    print(' ' * 15, end='')
    for genre in data_genres:
        print(f"{genre:^11}", end='')

    print(f"\n{'Precisions:':<15}", end='')
    for genre in data_genres:
        index = data_genres[genre]
        print(f"{precisions[index]:^11.2f}", end='')

    print(f"\n{'Recalls:':<15}", end='')
    for genre in data_genres:
        index = data_genres[genre]
        print(f"{recalls[index]:^11.2f}", end='')

    print(f"\n{'F-Scores:':<15}", end='')
    for genre in data_genres:
        index = data_genres[genre]
        print(f"{f_scores[index]:^11.2f}", end='')

    print(f"\n{'Accuracy:':<15}{accuracy * 100:.2f}%")

def custom_category():
    custom_path = input("Insert the absolute path of your song:")

    copy_data = np.load("features.npy")
    ccf(custom_path)
    pca()

    data = np.load("./pca.npy")
    data_labels = np.load("./labels.npy")

    train_data = data[:-1]
    train_labels = data_labels

    test_data = data[-1:]

    custom_predictions = gnb(train_data, train_labels, test_data)

    data_genres = {
        "blues": 0, "classical": 1, "country": 2,
        "disco": 3, "hiphop": 4, "jazz": 5, "metal": 6,
        "pop": 7, "reggae": 8, "rock": 9
    }

    index_to_genre = {index: genre for genre, index in data_genres.items()}

    genre_index = np.argmax(custom_predictions, axis=0)
    if isinstance(genre_index, np.ndarray):
        genre_index = genre_index.item()

    predicted_genre = index_to_genre.get(genre_index, "Unknown genre")
    print(f"Predicted genre: {predicted_genre}")

    np.save("features.npy", copy_data)


custom_category()

print("Statistics based on 80:20 test distribution:")
train_data, train_labels, test_data, true_labels, dataset = split_data(80)
predictions = gnb(train_data, train_labels, test_data)
computing_statistics(true_labels,predictions)



