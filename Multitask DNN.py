from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np
import pandas as pd
import deepchem as dc

#data processing
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
mt_df.to_csv("chembl_kkb_multi_task_data_083021.csv.gz", index=False, compression="gzip")

#model training
df = pd.read_csv("chembl_kkb_multi_task_data_083021.csv.gz")
FP_SIZE=1024
RADIUS=2
def calc_fp(smiles, fp_size, radius):
    """
    calcs morgan fingerprints as a numpy array.
    """
    mol = Chem.MolFromSmiles(smiles)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size)
    a = np.zeros((0,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, a)
    return a

descs = [calc_fp(smi, FP_SIZE, RADIUS) for smi in df["Canonical_Smiles"].values]
descs = np.asarray(descs, dtype=np.float32)

loader = dc.data.NumpyDataset(X=descs, y=df.iloc[:, 1:].values, n_tasks=342, ids=df.Canonical_Smiles)

splitter = dc.splits.RandomStratifiedSplitter()
train, valid, test = splitter.train_valid_test_split(loader)

model = dc.models.fcnet.MultitaskClassifier(342,
    1024,
    layer_sizes=[2000, 500],
    dropouts=[.25, .25],
    weight_init_stddevs=[.02, .02],
    bias_init_consts=[1., 1.],
    learning_rate=.0003,
    weight_decay_penalty=.0001,
    batch_size=100,
    seed=123,
    verbosity="high")

model.fit(train, nb_epoch=100)
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, mode="classification", classification_handling_mode="direct")
train_scores = model.evaluate(train, [metric])
valid_scores = model.evaluate(valid, [metric])
test_scores = model.evaluate(test, [metric])



