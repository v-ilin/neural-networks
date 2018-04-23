import csv

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, LabelEncoder


string_features_indices = [1, 3, 5, 6, 7, 8, 9, 13]
number_features_indices = [0, 2, 4, 10, 11, 12]


def replace_string_missings(data):
    for index in string_features_indices:
        label_encoder = LabelEncoder()
        strings = [row[index] for row in data]
        label_encoder.fit(strings)

        try:
            index_of_missing_char = list(label_encoder.classes_).index(" ?")
        except ValueError:
            continue

        transformed_strings = label_encoder.transform(strings)

        imp = Imputer(index_of_missing_char, strategy="most_frequent")
        transformed_strings_wo_missing = imp.fit_transform(transformed_strings.reshape(-1, 1))
        strings_wo_missings = label_encoder.inverse_transform(transformed_strings_wo_missing.reshape(1, -1)[0].astype(int))

        for i, row in enumerate(X_train):
            row[index] = strings_wo_missings[i]


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

replace_string_missings(X_train)

sgd_classifier = linear_model.SGDClassifier()
sgd_classifier.fit(X_train, y_train)

a = ""

