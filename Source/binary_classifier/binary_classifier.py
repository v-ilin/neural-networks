import csv
import copy
import numpy as np
from numpy import array

from sklearn import linear_model, model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer, LabelEncoder

string_features_indices = [1, 3, 5, 6, 7, 8, 9, 13]
string_features_label_encoders = dict()


def replace_strings(source):
    destination = copy.deepcopy(source)

    for index in string_features_indices:
        strings = [row[index] for row in source]

        if index in string_features_label_encoders:
            label_encoder = string_features_label_encoders[index]
        else:
            label_encoder = LabelEncoder()
            label_encoder.fit(strings)

            string_features_label_encoders[index] = label_encoder

        try:
            index_of_missing_char = list(label_encoder.classes_).index(" ?")
        except ValueError:
            index_of_missing_char = -1

        transformed_strings = label_encoder.transform(strings)

        if index_of_missing_char == -1:
            imp = Imputer(index_of_missing_char, strategy="most_frequent")
            transformed_strings_wo_missing = imp.fit_transform(transformed_strings.reshape(-1, 1))

            for i, row in enumerate(destination):
                row[index] = transformed_strings_wo_missing[i][0]
        else:
            for i, row in enumerate(destination):
                row[index] = transformed_strings[i]

    return array(destination).astype(float)


def get_source_data():
    with open('dataset/train2.csv', encoding='utf8') as source_data_file:
        csv_reader = csv.reader(source_data_file)

        X = []
        y = []

        for row in csv_reader:
            X.append(row[0:14])
            y.append(row[14])

        return X, y


def get_test_data():
    with open('dataset/test2.csv', encoding='utf8') as test_data_file:
        return list(csv.reader(test_data_file))


def log_predictions(predictions):
    with open('dataset/predictions.txt', encoding='utf8', mode='w+') as predictionsFile:
        for prediction in predictions:
            predictionsFile.write("%s\n" % int(prediction))


X, y = get_source_data()

X_train = replace_strings(X)
X_test = replace_strings(get_test_data())

y_train = array(y).astype(float)

cv = model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

classifier = linear_model.LogisticRegressionCV(cv=cv, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=30, multi_class='ovr', n_jobs=1,
          penalty='l1', random_state=0, solver='liblinear', tol=0.0001,
          verbose=0)

classifier.fit(X_train, y_train)
test_predictions = classifier.predict(X_test)

log_predictions(test_predictions)
#
# sgd_parameters_grid = {
#     'loss':[
#         'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron',
#         'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'
#     ],
#     'penalty': ['none', 'l2', 'l1', 'elasticnet'],
#     'alpha': np.linspace(0.0001, 0.001, num=10),
#     'max_iter': range(5, 10)
# }
#
# log_regression_parameters_grid = {
#     'penalty': ['l2', 'l1'],
#     'solver' : ['liblinear', 'saga'],
#     'max_iter': range(1, 70)
# }
#
# cv = model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# cv.get_n_splits(X_train, y_train)
#
# grid_cv = GridSearchCV(classifier, scoring='accuracy', cv = cv)
# grid_cv.fit(X_train, y_train)
# test_predictions = grid_cv.predict(X_test)
#
# log_predictions(test_predictions)
#
# print(str(grid_cv.best_estimator_))
# print(str(grid_cv.best_score_))
