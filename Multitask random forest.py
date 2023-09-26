import pandas as pd
import deepchem as dc
from sklearn.ensemble import RandomForestClassifier
import random


## Split
KFold = dc.splits.RandomSplitter()
KFold2 = KFold.k_fold_split(dataset, k=10)


## MTRF
for train, test in KFold2:
    trainX = pd.DataFrame(train.X)
    trainy = pd.DataFrame(train.y)
    testX = pd.DataFrame(test.X)
    testy = pd.DataFrame(test.y)
    
    RF1 = RandomForestClassifier(n_estimators=100,
                                   max_features="auto",
                                   n_jobs=4,
                                   random_state=random.seed(3),
                                   verbose=1)
    RF1.fit(trainX, trainy)


