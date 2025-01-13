import time
# for numerical computations
import numpy as np
# for data manipulation
import pandas as pd
# for splitting datasets and performing cross-validation
from sklearn.model_selection import train_test_split, StratifiedKFold
# for evaluating model performance
from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# data is loaded from house-votes-84.data, and column names are specified
data_path = 'data/house-votes-84.data'
columns = [
    "party",
    "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution", "physician-fee-freeze",
    "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras",
    "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending",
    "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"
]
data = pd.read_csv(data_path, header=None, names=columns)


def preprocess_data(unprocessed_data, processing_strategy):
    processed_data = unprocessed_data.copy()

    # transform 'y', 'n', '?' into 1, 0, NaN or 2
    processed_data = processed_data.replace({'y': 1, 'n': 0, '?': np.nan if processing_strategy == 1 else 2})

    # transform NaN into the most frequent value in each column
    if processing_strategy == 1:
        for column in processed_data.columns[1:]:
            # mode() is a function in pandas that returns the most frequent values in a column
            most_frequent = processed_data[column].mode()[0]
            # fillna() is a pandas function that replaces all NaN values in a column with a given value
            processed_data[column] = processed_data[column].fillna(most_frequent)

    return processed_data


# train a Naive Bayes classifier, calculate probabilities
def train_naive_bayes(training_features, target_labels, lambda_smoothing):
    # calculates the number of distinct classes (labels)
    n_classes = len(np.unique(target_labels))
    # counts how many samples belong to each class
    # np.bincount() counts the occurrence of each integer value in target_labels
    class_counts = np.bincount(target_labels)
    # Class Prior Probability is the probability of each class in the dataset
    # lambda_smoothing ensures that no probability is zero, even if a class appears less frequently
    class_priors = (class_counts + lambda_smoothing) / (len(target_labels) + n_classes * lambda_smoothing)

    # store the conditional probabilities of the features given the class
    feature_probs = {}
    # for each class calculate the conditional probability for each feature
    for c in range(n_classes):
        # extracts all the feature rows from training_features that belong to class c
        x_c = training_features[target_labels == c]
        # exclude feature values equal to 2
        x_c_filtered = np.where(x_c == 2, 0, x_c)
        # np.sum(x_c, axis=0) gives the sum of the values for each feature in the class c
        # x_c.shape[0] returns the number of samples in class c
        # calculates the conditional probability for each feature in the dataset given the class c
        feature_probs[c] = ((np.sum(x_c_filtered, axis=0) + lambda_smoothing) /
                            (x_c.shape[0] + n_classes * lambda_smoothing))

    return class_priors, feature_probs


# predicts the class labels for the given data x based on the trained Naive Bayes model
def predict_naive_bayes(x, class_priors, feature_probs):
    log_probs = []
    # iterates over each class to calculate the log-probability for that class
    for c in range(len(class_priors)):
        # ensures that no feature probability is  0 or 1 which could cause issues with the logarithm
        safe_feature_probs = np.clip(feature_probs[c], 1e-9, 1 - 1e-9)
        # calculates the log-probability of class c for each sample in x
        # np.log(class_priors[c]) gives the log-probability of the class c
        # the rest calculates the log-likelihood of the features given the class
        log_prob = (np.log(class_priors[c]) + np.sum(x * np.log(safe_feature_probs)
                                                     + (1 - x) * np.log(1 - safe_feature_probs), axis=1))
        log_probs.append(log_prob)

    # combines the log-probabilities for each class into array using np.vstack(log_probs), then transposes it with .T
    # np.argmax(..., axis=1) finds the class with the highest log-probability for each sample
    # predicting the class label
    return np.argmax(np.vstack(log_probs).T, axis=1)


# evaluates the Naive Bayes model on a given dataset
def evaluate_model(evaluation_data, lambda_smoothing):
    # split into features and target
    # extracts the feature columns and converts them to integers
    x = evaluation_data.iloc[:, 1:].astype(int).values
    # converts the target labels (party) into numeric values
    y = evaluation_data['party'].map({'republican': 0, 'democrat': 1}).values

    # splits the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    # trains the Naive Bayes model using the training data
    class_priors, feature_probs = train_naive_bayes(x_train, y_train, lambda_smoothing)
    # generates predictions for the training set
    train_predictions = predict_naive_bayes(x_train, class_priors, feature_probs)
    # calculates the accuracy of the predictions on the training data
    train_accuracy = accuracy_score(y_train, train_predictions) * 100

    # 10-fold cross-validation
    # data is split into 10 folds using StratifiedKFold to ensure that each fold has a representative class distribution
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    accuracies = []
    cross_val_accuracies = []
    # for each fold, the model is trained on the training data and tested on the validation data
    # with the accuracy calculated for each fold
    for fold, (train_index, val_index) in enumerate(skf.split(x_train, y_train), start=1):
        # x_fold_train and y_fold_train are features and labels for training in this fold
        # x_fold_val and y_fold_val are features and labels for validation in this fold
        x_fold_train, x_fold_val = x_train[train_index], x_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

        # trains the Naive Bayes model on the training subset for the current fold
        class_priors, feature_probs = train_naive_bayes(x_fold_train, y_fold_train, lambda_smoothing)
        # predicts the labels of the validation subset using the trained model
        val_predictions = predict_naive_bayes(x_fold_val, class_priors, feature_probs)

        accuracies.append(accuracy_score(y_fold_val, val_predictions) * 100)
        (cross_val_accuracies.append
         (f"    Accuracy Fold {fold}: {accuracy_score(y_fold_val, val_predictions) * 100:.2f}%"))

    cross_val_mean = np.mean(accuracies)
    cross_val_std = np.std(accuracies)

    # after training the model is evaluated on the test set and the accuracy is calculated
    class_priors, feature_probs = train_naive_bayes(x_test, y_test, lambda_smoothing)
    test_predictions = predict_naive_bayes(x_test, class_priors, feature_probs)
    test_accuracy = accuracy_score(y_test, test_predictions) * 100

    return {
        "train_accuracy": train_accuracy,
        "cross_val_accuracies": cross_val_accuracies,
        "cross_val_mean": cross_val_mean,
        "cross_val_std": cross_val_std,
        "test_accuracy": test_accuracy
    }


start_time = time.time()

strategy = int(input('0 or 1: '))
data_strategy = []
if strategy == 0:
    # Evaluate for strategy 0 (treat '?' as third value
    data_strategy = preprocess_data(data, processing_strategy=0)
elif strategy == 1:
    # Evaluate for strategy 1 (replace '?' with most frequent value
    data_strategy = preprocess_data(data, processing_strategy=1)

results_strategy = evaluate_model(data_strategy, lambda_smoothing=1.0)

print(f"1. Train Set Accuracy:\n    Accuracy: {results_strategy['train_accuracy']:.2f}%")
print("\n10-Fold Cross-Validation Results:")
for fold_result in results_strategy['cross_val_accuracies']:
    print(fold_result)
print(f"\n    Average Accuracy: {results_strategy['cross_val_mean']:.2f}%")
print(f"    Standard Dev: {results_strategy['cross_val_std']:.2f}%")
print(f"\n 2.Test Accuracy:\n   Accuracy: {results_strategy['test_accuracy']:.2f}%")

end_time = time.time()
execution_time = round(end_time - start_time, 4)
print(execution_time)
