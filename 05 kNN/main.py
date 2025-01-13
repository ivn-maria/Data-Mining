import time
# for numerical computations
import numpy as np
# for data manipulation
import pandas as pd
# for splitting datasets and performing cross-validation
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
# for counting occurrences of labels in the k-nearest neighbors
from collections import Counter
# for plotting the result
import matplotlib.pyplot as plt


def load_iris_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    raw_data = pd.read_csv(url, header=None, names=column_names)
    return raw_data


def normalize_data(normalizing_data):
    # MinMaxScaler is a preprocessing utility from sklearn that scales each feature independently
    # it transforms data such that each feature value is scaled to fall within the range [0,1]
    scaler = MinMaxScaler()
    # data.iloc[:, :-1] selects all columns except the last one
    features = normalizing_data.iloc[:, :-1]
    # fit computes the minimum and maximum values for each feature
    # transform applies the normalization formula to scale all feature values
    normalizing_features = scaler.fit_transform(features)
    # pd.DataFrame converts the normalized NumPy array back into a Pandas DataFrame
    # data.iloc[:, -1] returns the label column as is, since it does not require scaling
    return pd.DataFrame(normalizing_features, columns=features.columns), normalizing_data.iloc[:, -1]


# calculates the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# predicts the class label for a given test point by finding the majority label among its k-nearest neighbors
def knn_predict(k_predict, train_features_predict, train_labels_predict, test_point):
    # calculates the Euclidean distance for each training point
    distances = np.array([euclidean_distance(train_point, test_point) for train_point in train_features_predict])
    # returns indices that would sort the distances in ascending order,
    # [:k_predict] selects the indices of the k-smallest distances (nearest neighbors)
    nearest_indices = np.argsort(distances)[:k_predict]
    # uses the indices of the nearest neighbors to extract their labels from the training set
    nearest_labels = train_labels_predict[nearest_indices]
    # counts occurrences of each label,
    # most_common(1) retrieves the most frequent label and its count, [0][0] extracts the label itself
    most_common_label = Counter(nearest_labels).most_common(1)[0][0]
    return most_common_label


# measures how well the kNN algorithm predicts labels on a given dataset
def knn_accuracy(k_accuracy, train_features_accuracy, train_labels_accuracy, test_features_accuracy,
                 test_labels_accuracy):
    correct_predictions = 0

    # iterates through all test points
    for test_point, true_label in zip(test_features_accuracy, test_labels_accuracy):
        # uses knn_predict to get predictions
        prediction = knn_predict(k_accuracy, train_features_accuracy, train_labels_accuracy, test_point)
        if prediction == true_label:
            correct_predictions += 1

    # calculates the fraction of correct predictions for accuracy
    return correct_predictions / len(test_labels_accuracy)


# evaluates the performance of kNN algorithm using 10-Fold Cross-Validation
def cross_validation_accuracy(k_validation, features_validation, labels_validation):
    # StratifiedKFold ensures that the distribution of labels in each fold matches the overall dataset's distribution
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    fold_results_validation = []

    for fold, (train_index, test_index) in enumerate(skf.split(features_validation, labels_validation), start=1):
        # split Data into Training (Data points from 9 folds) and Testing (Data points from the remaining fold) Sets
        train_features_validation, test_features_validation = (
            features_validation[train_index], features_validation[test_index])
        train_labels_validation, test_labels_validation = labels_validation[train_index], labels_validation[test_index]

        # train and evaluate by using the training set to make predictions on the testing set with knn_accuracy
        accuracy = knn_accuracy(k_validation, train_features_validation, train_labels_validation,
                                test_features_validation, test_labels_validation)
        # store Fold Accuracy
        accuracies.append(accuracy)
        fold_results_validation.append(f"    Accuracy Fold {fold}: {accuracy * 100:.2f}%")

    # computes mean and standard deviation of accuracy across folds.
    return np.mean(accuracies), np.std(accuracies), fold_results_validation


start_time = time.time()

# loads and normalizes the dataset
data = load_iris_dataset()
normalized_features, labels = normalize_data(data)

# splits the data into training (80%) and test (20%) sets while maintaining class distribution
train_features, test_features, train_labels, test_labels = train_test_split(
    normalized_features.values, labels.values, test_size=0.2, stratify=labels, random_state=42
)

# records accuracy on the training set, cross-validation accuracy (with standard deviation), accuracy on the test set
# prints accuracy for a user-specified k on the training set, cross-validation, and test set
k = int(input('k: '))

train_accuracy = knn_accuracy(k, train_features, train_labels, train_features, train_labels)
cross_val_accuracy, cross_val_stddev, fold_results = cross_validation_accuracy(k, train_features, train_labels)
test_accuracy = knn_accuracy(k, train_features, train_labels, test_features, test_labels)

print(f"1. Train Set Accuracy:\n    Accuracy: {train_accuracy * 100:.2f}%")
print("\n2. 10-Fold Cross-Validation Results:")
for fold_result in fold_results:
    print(fold_result)
print(f"\n    Average Accuracy: {cross_val_accuracy * 100:.2f}%")
print(f"    Standard Deviation: {cross_val_stddev * 100:.2f}%")
print(f"\n3. Test Set Accuracy:\n    Accuracy: {test_accuracy * 100:.2f}%")

end_time = time.time()
execution_time = round(end_time - start_time, 4)
print(execution_time)

start_time = time.time()

# experiment with different k values from 1 to 20
k_values = range(1, 21)
train_accuracies = []
cross_val_accuracies = []
cross_val_stddevs = []
cross_val_folds = []
test_accuracies = []

# records accuracy on the training set, cross-validation accuracy (with standard deviation), accuracy on the test set
for k in k_values:
    train_accuracy = knn_accuracy(k, train_features, train_labels, train_features, train_labels)
    cross_val_accuracy, cross_val_stddev, fold_results = cross_validation_accuracy(k, train_features, train_labels)
    test_accuracy = knn_accuracy(k, train_features, train_labels, test_features, test_labels)

    train_accuracies.append(train_accuracy)
    cross_val_accuracies.append(cross_val_accuracy)
    cross_val_stddevs.append(cross_val_stddev)
    cross_val_folds.append(fold_results)
    test_accuracies.append(test_accuracy)

# visualizes how accuracy varies with k for training, cross-validation, and test sets
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accuracies, label="Train Accuracy")
plt.plot(k_values, cross_val_accuracies, label="Cross-Validation Accuracy")
plt.plot(k_values, test_accuracies, label="Test Accuracy")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")
plt.legend()
plt.grid()
plt.show()

end_time = time.time()
execution_time = round(end_time - start_time, 4)
print(execution_time)
