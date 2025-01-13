import math
import random
# for data manipulation and statistical operations
import numpy as np
import pandas as pd
# encodes categorical data into numerical labels
from sklearn.preprocessing import LabelEncoder


# measures the randomness or impurity in a dataset
def _entropy(labels):
    label_counts = {}
    for label in labels:
        # if the label exists in label_counts, its count is incremented by 1
        # if it does not exist, it is initialized with a count of 0 and incremented to 1
        label_counts[label] = label_counts.get(label, 0) + 1

    entropy = 0
    total = len(labels)

    # calculate the probability and entropy contribution of each label
    # for each label, calculates its proportion and computes −plog2(p), summing this for all labels
    for count in label_counts.values():
        prob = count / total
        entropy -= prob * math.log2(prob)

    return entropy


# calculates how much uncertainty (entropy) is reduced by splitting data on a feature
def _information_gain(labels, feature_values):
    total_entropy = _entropy(labels)
    feature_entropy = 0
    total_samples = len(labels)

    # for each unique feature value splits the labels corresponding to this value
    # and computes the weighted entropy of the subsets
    unique_values = set(feature_values)
    for value in unique_values:
        # splits the dataset based on the unique values of that feature
        subset_indices = [i for i in range(len(feature_values)) if feature_values[i] == value]
        subset_labels = [labels[i] for i in subset_indices]
        subset_entropy = _entropy(subset_labels)
        weight = len(subset_indices) / total_samples
        feature_entropy += weight * subset_entropy

    # information gain = total entropy − feature entropy
    return total_entropy - feature_entropy


# finds the most frequent label in a dataset
def _most_common_label(labels):
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    return max(label_counts, key=label_counts.get)


# splits data into two parts based on a feature’s value
def _split_data(x, y, feature_index, feature_value):
    x_left, y_left = [], []
    x_right, y_right = [], []

    for i in range(len(x)):
        if x[i][feature_index] == feature_value:
            x_left.append(x[i])
            y_left.append(y[i])
        else:
            x_right.append(x[i])
            y_right.append(y[i])

    return x_left, y_left, x_right, y_right


# replaces missing values (?) with the mode (most frequent value) of the feature
def _impute_missing_values(x):
    # check if x is empty or has no features (columns)
    if isinstance(x, np.ndarray):
        if x.size == 0 or x.shape[1] == 0:
            return
    # check if x is an empty list
    elif not x:
        return

    for feature in range(len(x[0])):
        # collect non-missing values for each feature
        feature_values = [row[feature] for row in x if row[feature] != '?']

        # Handle the case where feature_values is empty
        if feature_values:
            mode = max(set(feature_values), key=feature_values.count)

            for row in x:
                if row[feature] == '?':
                    row[feature] = mode


class ID3Tree:
    # max_depth is the maximum depth of the tree (limits overfitting)
    # min_samples_leaf is the minimum samples required in a leaf node
    # min_gain is the minimum information gain required to split a node
    def __init__(self, max_depth=5, min_samples_leaf=3, min_gain=0.01, pruning_type='0'):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.tree = None
        self.pruning_type = pruning_type

    # tree construction based on the training data (x and y)
    def _build_tree(self, x, y, depth=0):
        # pre-pruning
        if self.pruning_type == '0' or self.pruning_type == '2':
            # stop if criteria are met: maximum depth reached, insufficient samples for splitting,
            # all labels are the same returns the most common label if criteria are met
            if (depth >= self.max_depth or
                    len(y) < 2 * self.min_samples_leaf or
                    len(set(y)) == 1):
                return _most_common_label(y)

        if ('0' in self.pruning_type or '2' in self.pruning_type) and 'N' in self.pruning_type:
            if depth >= self.max_depth:
                return _most_common_label(y)

        if ('0' in self.pruning_type or '2' in self.pruning_type) and 'K' in self.pruning_type:
            if len(y) < 2 * self.min_samples_leaf:
                return _most_common_label(y)

        if ('0' in self.pruning_type or '2' in self.pruning_type) and 'G' in self.pruning_type:
            if len(set(y)) == 1:
                return _most_common_label(y)

        # find the best feature
        best_gain = 0
        best_feature = None
        best_value = None

        # loops through all features and their unique values
        # splits the data and calculates information gain
        # records the best feature and value with the highest gain
        for feature in range(len(x[0])):
            unique_values = set(row[feature] for row in x)
            for value in unique_values:
                x_left, y_left, x_right, y_right = _split_data(x, y, feature, value)

                # check for minimum number of samples
                if (len(y_left) < self.min_samples_leaf or
                        len(y_right) < self.min_samples_leaf):
                    continue

                # records the feature and value with the highest information gain that meets the min_gain threshold
                gain = _information_gain(y, [row[feature] for row in x])

                if gain > best_gain and gain >= self.min_gain:
                    best_gain = gain
                    best_feature = feature
                    best_value = value

        # if no valid split is found
        if best_feature is None:
            return _most_common_label(y)

        # splits the data on the best feature and value and recursively builds the left and right subtrees
        x_left, y_left, x_right, y_right = _split_data(x, y, best_feature, best_value)

        tree = {
            'feature': best_feature,
            'value': best_value,
            'left': self._build_tree(x_left, y_left, depth + 1),
            'right': self._build_tree(x_right, y_right, depth + 1)
        }

        # post-pruning
        if self.pruning_type == '1' or self.pruning_type == '2':
            tree = self._reduced_error_pruning(tree, x, y)

        return tree

    # handles missing values, calls _build_tree to recursively construct the decision tree
    def fit(self, x, y):
        _impute_missing_values(x)

        self.tree = self._build_tree(x, y)
        return self

    def _reduced_error_pruning(self, tree, x, y):
        # split the data for validation
        x_train, x_val, y_train, y_val = stratified_split(x, y, test_size=0.2)

        # perform pruning by checking if pruning leads to a better error
        before_error = self._calculate_error(tree, x_val, y_val)
        error_after_pruning_left = self._calculate_error(tree['left'], x_val, y_val)
        error_after_pruning_right = self._calculate_error(tree['right'], x_val, y_val)

        # if pruning improves error, return the most common label
        if before_error > (error_after_pruning_left + error_after_pruning_right):
            # prune subtree by replacing it with the most common label
            return _most_common_label(y)

        return tree

    def _calculate_error(self, tree, x, y):
        # check if validation labels are empty
        if len(y) == 0:
            # no error if no labels are available
            return 0

        predictions = self.predict(x)
        # computes the error rate by counting how many predictions differ from the true labels (p != t)
        # and dividing that by the total number of labels (len(y))
        # the result is the fraction of incorrect predictions (error rate)
        return sum(p != t for p, t in zip(predictions, y)) / len(y)

    # for each sample traverses the tree based on feature values,
    # stops when it reaches a leaf node and outputs its label
    def predict(self, x):
        # handle missing values
        _impute_missing_values(x)

        predictions = []
        for sample in x:
            predictions.append(self._predict_sample(sample))

        return predictions

    def _predict_sample(self, sample):
        # the root of the decision tree
        node = self.tree
        # continues as long as node is a dictionary, meaning we are still in an internal node (not a leaf node)
        while isinstance(node, dict):
            feature = node['feature']
            value = node['value']

            if sample[feature] == value:
                node = node['left']
            else:
                node = node['right']

        return node


# data preparation (stratified data split)
# splits data into train and test sets while maintaining label proportions
# ensures balanced class distribution in both sets
def stratified_split(x, y, test_size=0.2, random_state=42):
    random.seed(random_state)

    # group indices by class
    class_indices = {}
    for idx, label in enumerate(y):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    train_indices, test_indices = [], []
    # split for each class
    for label, indices in class_indices.items():
        num_test = int(len(indices) * test_size)
        # shuffle indices
        random.shuffle(indices)
        test_indices.extend(indices[:num_test])
        train_indices.extend(indices[num_test:])

    # create datasets
    x_train = [x[idx] for idx in train_indices]
    y_train = [y[idx] for idx in train_indices]
    x_test = [x[idx] for idx in test_indices]
    y_test = [y[idx] for idx in test_indices]

    return x_train, x_test, y_train, y_test


# 10-fold cross-validation
# splits data into 10 subsets (folds)
# uses 9 folds for training and 1 for validation in each iteration
# computes accuracy for each fold and reports mean accuracy and standard deviation of accuracies
def cross_validation(x, y, pruning_type, n_splits=10):
    fold_size = len(x) // n_splits
    accuracies = []

    for i in range(n_splits):
        # split the data
        start = i * fold_size
        end = (i + 1) * fold_size

        x_val = x[start:end]
        y_val = y[start:end]

        x_train = x[:start] + x[end:]
        y_train = y[:start] + y[end:]

        # train and test
        tree = ID3Tree(pruning_type=pruning_type)
        tree.fit(x_train, y_train)
        predictions = tree.predict(x_val)

        # accuracy
        accuracy = sum(p == t for p, t in zip(predictions, y_val)) / len(y_val)
        accuracies.append(accuracy * 100)

    return accuracies


def run_experiment(pruning_type):
    # load dataset
    column_names = [
        'Class', 'Age', 'Menopause', 'Tumor-Size', 'Inv-Nodes',
        'Node-Caps', 'Deg-Malig', 'Breast', 'Breast-Quad', 'Irradiat'
    ]
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data'
    data = pd.read_csv(url, names=column_names)
    data.replace('?', pd.NA, inplace=True)
    data.dropna(inplace=True)

    # encode categorical data
    for col in data.columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    # split data into features and labels
    x = data.drop('Class', axis=1).values
    y = data['Class'].values

    x_train, x_test, y_train, y_test = stratified_split(x, y)

    tree = ID3Tree(pruning_type=pruning_type)
    # train
    tree.fit(x_train, y_train)
    # train set accuracy
    train_predictions = tree.predict(x_train)
    train_accuracy = sum(p == t for p, t in zip(train_predictions, y_train)) / len(y_train) * 100
    # cross-validation
    cv_accuracies = cross_validation(x_train, y_train, pruning_type)
    # test set accuracy
    test_predictions = tree.predict(x_test)
    test_accuracy = sum(p == t for p, t in zip(test_predictions, y_test)) / len(y_test) * 100

    # output results
    print(f"1. Train Set Accuracy:\n\tAccuracy: {train_accuracy:.2f}%")

    print("\n10-Fold Cross-Validation Results:")
    for i, acc in enumerate(cv_accuracies, 1):
        print(f"\tAccuracy Fold {i}: {acc:.2f}%")

    print(f"\tAverage Accuracy: {sum(cv_accuracies)/len(cv_accuracies):.2f}%")
    print(f"\tStandard Deviation: {np.std(cv_accuracies):.2f}%")

    print(f"\n2. Test Set Accuracy:\n\tAccuracy: {test_accuracy:.2f}%")


def main():
    pruning_type = input("Enter pruning type: ").strip().upper()
    run_experiment(pruning_type)


if __name__ == "__main__":
    main()
