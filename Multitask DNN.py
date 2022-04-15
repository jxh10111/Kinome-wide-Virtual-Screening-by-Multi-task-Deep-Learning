from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np
import pandas as pd
import deepchem as dc


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



