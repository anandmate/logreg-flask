import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

import pickle

insure = pd.read_csv('F:\My Data Science\Data Analytics\insurance2.csv')

# print(insure.head())
# print(insure.isnull().sum())

# Dividing in Features and target
X = insure.iloc[:, :7]
y = insure.iloc[:, 7:]

# print(X.shape)
# print(y.shape)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# create a model
logreg = LogisticRegression()

logreg.fit(X_train, y_train.values.ravel())


# prediction
# y_pred = logreg.predict(X_test)
#
#
# #evaluation
# cmax = confusion_matrix(y_test, y_pred)
# print(cmax)
#
# report = classification_report(y_test, y_pred)
# print(report)


# Saving the model
pickle.dump(logreg, open('logreg.pkl', 'wb'))
