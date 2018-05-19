import os.path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import seaborn as sns
import pandas as pd
import matplotlib as plt

dataset_dir = 'dataset'

train_filenames = [
    '0_train.txt',
    '1_train.txt',
    '2_train.txt',
    '3_train.txt',
    '4_train.txt',
    '5_train.txt',
    '6_train.txt'
]

train_filepaths = [os.path.join(dataset_dir, filename) for filename in train_filenames]
val_filepath = os.path.join(dataset_dir, "test.txt")

train_data = []
y = []

for cls_idx, train_filename in enumerate(train_filepaths):
    with open(train_filename, 'r', encoding='utf-8') as train_data_file:
        train_data_cls = train_data_file.read().splitlines()
        train_data.extend(train_data_cls)

        y_cls = [cls_idx] * len(train_data_cls)
        y.extend(y_cls)

count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(train_data)

tf_idf_transformer = TfidfTransformer()
X = tf_idf_transformer.fit_transform(X_counts)

clf = LinearSVC()
clf.fit(X, y)

with open(val_filepath, 'r', encoding='utf-8') as val_data_file:
    val_data = val_data_file.read().splitlines()
    X_test = count_vect.transform(val_data)
    predictions = clf.predict(X_test)

    for i in predictions:
        print(str(i))

# models = [
#     RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
#     LinearSVC(),
#     MultinomialNB(),
#     LogisticRegression(random_state=0),
# ]
# CV = 5
# cv_df = pd.DataFrame(index=range(CV * len(models)))
# entries = []
# for model in models:
#   model_name = model.__class__.__name__
#   accuracies = cross_val_score(model, X, y, scoring='accuracy', cv=CV)
#   for fold_idx, accuracy in enumerate(accuracies):
#     entries.append((model_name, fold_idx, accuracy))
# cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
#
# sns.boxplot(x='model_name', y='accuracy', data=cv_df)
# sns.stripplot(x='model_name', y='accuracy', data=cv_df,
#               size=8, jitter=True, edgecolor="gray", linewidth=2)
# plt.pyplot.show()
a = ''


