import csv
import copy
from numpy import array

from sklearn import linear_model
from sklearn.model_selection import train_test_split
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


X, y = get_source_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_floats = replace_strings(X_train)
X_test_floats = replace_strings(X_test)

y_train_floats = array(y_train).astype(float)
y_test_floats = array(y_test).astype(float)

sgd_classifier = linear_model.SGDClassifier()
sgd_classifier.fit(X_train_floats, y_train_floats)
accuracy = sgd_classifier.score(X_test_floats, y_test_floats)

print(str(accuracy))
