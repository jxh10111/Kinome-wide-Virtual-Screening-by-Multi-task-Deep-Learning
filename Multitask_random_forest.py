# Part 1: Training dataset input Processing
import os
import numpy as np
import pandas as pd
from rdkit import Chem
import deepchem as dc
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import random
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
import joblib
import seaborn as sns
import dill
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdMolDescriptors
import json

kkb_chembl_file = "kkb_chembl_standardizedAggregation_geq15_061916.csv.gz"
df = pd.read_csv(kkb_chembl_file)

df["active"] = df.kinasemodel6.apply(lambda x: 1 if x is not np.nan else 0)
def gen_dict(group):
    return {tid: act for tid, act in zip(group["AssayTarget"], group["active"])}

group = df.groupby("Canonical_Smiles")
temp = pd.DataFrame(group.apply(gen_dict))
mt_df = pd.DataFrame(temp[0].tolist())
mt_df["Canonical_Smiles"] = temp.index
mt_df = mt_df.where((pd.notnull(mt_df)), -1)

structs = df[["Canonical_Smiles"]].drop_duplicates("Canonical_Smiles")
structs["romol"] = structs.apply(lambda row: Chem.MolFromSmiles(row["Canonical_Smiles"]), axis=1)
structs = structs.dropna()
del structs["romol"]

mt_df = pd.merge(structs, mt_df, how="inner", on="Canonical_Smiles")
mt_df = mt_df.replace(-1, 0)
k = mt_df.sum()
k.drop("Canonical_Smiles", inplace=True)
k = k.astype(float)

todrop = k[k < 15].index
mt_df = mt_df.drop(todrop, axis=1)

mt_df.to_csv("chembl_kkb_multi_task_data_083021.csv.gz", index=False, compression='gzip')

mt_df = pd.read_csv ("chembl_kkb_multi_task_data_083021.csv.gz")
loader = dc.data.CSVLoader(tasks=mt_df.columns.values[1:].tolist(),
                           featurizer=dc.feat.CircularFingerprint(size=1024, radius=2),
                           smiles_field="Canonical_Smiles")

#os.makedirs("dc_test", exist_ok=True)
dataset = loader.featurize("chembl_kkb_multi_task_data_083021.csv.gz", data_dir="dc_test")

# Part 2: Multitask Random Forest

KFold = dc.splits.RandomSplitter()
KFold2 = KFold.k_fold_split(dataset, k=10)

tprs = []
base_fpr = np.linspace(0, 1, 101)
final_roc = []
mark=0
for train, test in KFold2:
    print(mark)
    file_name = "ROC_"+ str(mark) +".png"
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
    model_name = "random_forest_"+str(mark)+".joblib"
    joblib.dump(RF1, model_name,compress=8)
    y_pred_probs = RF1.predict_proba(testX)
    y_pred_probs2  = np.array(y_pred_probs)
    y_pred_probs3 = y_pred_probs2[:,:,1]
    y_pred_probs4 = np.transpose(y_pred_probs3)
    testy2 = np.array(testy)
    
    n_classes = testy2.shape[1]
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(testy2[:,i], y_pred_probs4[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    roc_auc2 =  pd.DataFrame(list(roc_auc.items()),index=dataset.tasks)
    roc_auc3 = roc_auc2.drop(0, axis=1) 
    roc_auc3.columns = ["ROC"]
    roc_file_name = "roc_auc_per_kinase_"+str(mark)+".csv"
    roc_auc3.to_csv(roc_file_name)
    del RF1

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(testy2.ravel(), y_pred_probs4.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    

    plt.plot(fpr["micro"], tpr["micro"], 'b', alpha=0.15,
             label="ROC curve (area = %0.2f)" % roc_auc["micro"])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(file_name)
    plt.show()
    tpr = interp(base_fpr, fpr["micro"], tpr["micro"])
    tpr[0] = 0.0
    tprs.append(tpr)
    final_roc.append(roc_auc["micro"])
    
    mark = mark+1

final_roc = np.array(final_roc)
