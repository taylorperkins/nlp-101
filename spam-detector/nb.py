import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier


data = pd.read_csv('../data/spambase.data').as_matrix()
np.random.shuffle(data)

# https://archive.ics.uci.edu/ml/datasets/Spambase
# The data for this project has the first 48 columns as features, and the last as
# the target. Spam: 0 | 1
X = data[:, :48]
Y = data[:, -1]

Xtrain = X[:-100, ]
Ytrain = Y[:-100, ]
Xtest = X[-100:, ]
Ytest = Y[-100:, ]

model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print(f"Classification rate for NB: {model.score(Xtest, Ytest)}")

model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print(f"Classification for AdaBoostClassifier: {model.score(Xtest, Ytest)}")



