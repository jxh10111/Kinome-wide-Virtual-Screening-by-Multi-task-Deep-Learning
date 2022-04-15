{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c91b4af",
   "metadata": {},
   "outputs": [],
   "source": [
    "from rdkit import Chem\n",
    "from rdkit.Chem import rdMolDescriptors\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import deepchem as dc\n",
    "\n",
    "\n",
    "df = pd.read_csv(\"/media/ballen/overflow/Downloads/chembl_kkb_multi_task_data_083021.csv.gz\")\n",
    "FP_SIZE=1024\n",
    "RADIUS=2\n",
    "def calc_fp(smiles, fp_size, radius):\n",
    "    \"\"\"\n",
    "    calcs morgan fingerprints as a numpy array.\n",
    "    \"\"\"\n",
    "    mol = Chem.MolFromSmiles(smiles)\n",
    "    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size)\n",
    "    a = np.zeros((0,), dtype=np.float32)\n",
    "    Chem.DataStructs.ConvertToNumpyArray(fp, a)\n",
    "    return a\n",
    "\n",
    "descs = [calc_fp(smi, FP_SIZE, RADIUS) for smi in df[\"Canonical_Smiles\"].values]\n",
    "descs = np.asarray(descs, dtype=np.float32)\n",
    "\n",
    "loader = dc.data.NumpyDataset(X=descs, y=df.iloc[:, 1:].values, n_tasks=342, ids=df.Canonical_Smiles)\n",
    "\n",
    "splitter = dc.splits.RandomStratifiedSplitter()\n",
    "train, valid, test = splitter.train_valid_test_split(loader)\n",
    "\n",
    "model = dc.models.fcnet.MultitaskClassifier(342,\n",
    "    1024,\n",
    "    layer_sizes=[2000, 500],\n",
    "    dropouts=[.25, .25],\n",
    "    weight_init_stddevs=[.02, .02],\n",
    "    bias_init_consts=[1., 1.],\n",
    "    learning_rate=.0003,\n",
    "    weight_decay_penalty=.0001,\n",
    "    batch_size=100,\n",
    "    seed=123,\n",
    "    verbosity=\"high\")\n",
    "\n",
    "model.fit(train, nb_epoch=100)\n",
    "metric = dc.metrics.Metric(dc.metrics.roc_auc_score, mode=\"classification\", classification_handling_mode=\"direct\")\n",
    "train_scores = model.evaluate(train, [metric])\n",
    "valid_scores = model.evaluate(valid, [metric])\n",
    "test_scores = model.evaluate(test, [metric])\n",
    "\n",
    "idg_df = pd.read_csv(\"/media/ballen/overflow/Downloads/IDG_Challenge_round1_pKd_fixed.csv.gz\")\n",
    "idg_descs = [calc_fp(smi, FP_SIZE, RADIUS) for smi in idg_df[\"Compound SMILES\"].values]\n",
    "idg_descs = np.asarray(idg_descs, dtype=np.float32)\n",
    "idg = dc.data.NumpyDataset(X=idg_descs, y=idg_df.iloc[:, 1:].values, n_tasks=342, ids=idg_df[\"Compound SMILES\"])\n",
    "idg_pred = model.predict(idg)\n",
    "idg_act = idg_pred[:,:,1]\n",
    "idg_act_df = pd.DataFrame(idg_act, columns=idg_df.columns[1:], index=idg_df[\"Compound SMILES\"])\n",
    "idg_act_df.to_csv(\"IDG_Challenge_round1_MTDNN_KinasePred.csv\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:deepchem] *",
   "language": "python",
   "name": "conda-env-deepchem-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
