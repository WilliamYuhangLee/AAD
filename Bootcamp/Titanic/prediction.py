from collections import Counter
import pandas as pd
import numpy as np
from sklearn import svm, tree, neighbors, neural_network, ensemble, linear_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix

# Preprocessing the data

# Raw data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# Outlier detection
def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than n outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers


# detect outliers from Age, SibSp and Parch
outliers_to_drop = detect_outliers(train_data, 2, ["Age", "SibSp", "Parch"])

# Drop outliers
train_data = train_data.drop(outliers_to_drop, axis=0).reset_index(drop=True)

# Print columns that have missing data
# print(train_data.columns[train_data.isna().any()].tolist())

# Drop columns that are not useful for now
train_data.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)
test_data.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

# Set primary key (and separate from the rest of columns)
train_data.set_index(keys=["PassengerId"], drop=True, inplace=True)
test_data.set_index(keys=["PassengerId"], drop=True, inplace=True)

# Set replace values for the missing data
train_nan_map = {'Age': train_data['Age'].mean(), 'Fare': train_data['Fare'].mean(), 'Embarked': train_data['Embarked'].mode()[0]}
test_nan_map = {'Age': test_data['Age'].mean(), 'Fare': test_data['Fare'].mean(), 'Embarked': test_data['Embarked'].mode()[0]}

# Replace missing data with set values
train_data.fillna(value=train_nan_map, inplace=True)
test_data.fillna(value=test_nan_map, inplace=True)

# Translate string values to int values
columns_map = {'Embarked': {'C': 0, 'Q': 1, 'S': 2}, 'Sex': {'male': 0, 'female': 1}}
train_data.replace(columns_map, inplace=True)
test_data.replace(columns_map, inplace=True)

# Separate x and y columns
X_train = train_data.loc[:, train_data.columns != 'Survived']
y_train = train_data.loc[:, 'Survived']

# Split a part of training data as test data (to avoid using test_data for testing)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=10)

# # Using Decision Tree Classifier from sci-kit learn to train
# tree_clf = tree.DecisionTreeClassifier()
# tree_clf.fit(X_train.values, y_train.values)
# print("Decision Tree:", tree_clf.score(X_test.values, y_test.values))
#
# # Using K Neighbors Classifier from sci-kit learn to train
# knn_clf = neighbors.KNeighborsClassifier()
# knn_clf.fit(X_train.values, y_train.values)
# print("K neighbors:", knn_clf.score(X_test.values, y_test.values))
#
# # Using Neural Network Classifier from sci-kit learn to train
# NN_clf = neural_network.MLPClassifier()
# NN_clf.fit(X_train.values, y_train.values)
# print("Neural Network:", NN_clf.score(X_test.values, y_test.values))
#
# # Using Random Forest Classifier from sci-kit learn to train
# RF_clf = ensemble.RandomForestClassifier()
# RF_clf.fit(X_train.values, y_train.values)
# print("Random Forest:", RF_clf.score(X_test.values, y_test.values))
#
# # Using Logistic Regression Classifier from sci-kit learn to train
# LR_clf = linear_model.LogisticRegression()
# LR_clf.fit(X_train.values, y_train.values)
# print("Logistic Regression:", LR_clf.score(X_test.values, y_test.values))

# Training

# Parameters Tuning
def svc_param_selection(x, y):
    Cs = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
    gammas = [1, 0.1, 0.01, 0.001, 0.0001]
    param_grid = [{"kernel": ["linear", "rbf"], "C": Cs, "gamma": gammas}]
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5)
    grid_search.fit(x, y)
    return grid_search.best_params_


# best_params = {"kernel": "linear", "C": 0.01}  # arbitrary parameters
best_params = svc_param_selection(X_train.values, y_train.values)
print("Best parameters detected:", best_params)

# Using SVM Classifier from sci-kit learn to train
svm_clf = svm.SVC()
svm_clf.set_params(**best_params)
svm_clf.fit(X_train.values, y_train.values)
print("SVM:", svm_clf.score(X_test.values, y_test.values))

# Using SVM to predict
y_pred = svm_clf.predict(X_test.values)
y_truth = y_test.values

# Get confusion matrix
tn, fp, fn, tp = confusion_matrix(y_truth, y_pred).ravel()
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

# Print Confusion matrix and precision rates
print("Confusion Matrix")
print(confusion_matrix(y_truth, y_pred, labels=[0, 1]))
print("")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)
print("")
print("False Positive Rate:", fpr)
print("False Negative Rate:", fnr)

# Prediction

# Real prediction on test_data
predictions = svm_clf.predict(test_data.values)
pred_df = pd.DataFrame(predictions, index=test_data.index, columns=['Survived'])
pred_df.to_csv('out/predictions.csv', header=True, sep=',')
