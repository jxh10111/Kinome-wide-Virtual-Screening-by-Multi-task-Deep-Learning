import numpy as np
import pandas as pd
from rdkit import Chem


input_file = "kkb_chembl_standardizedAggregation_geq15_061916.csv.gz"
output_file = "chembl_kkb_multi_task_data.csv.gz"


class DataProcessor:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.df = None
        self.mt_df = None
        self.structs = None

    def read_data(self):
        self.df = pd.read_csv(self.input_file)

    def add_active_column(self):
        self.df["active"] = self.df.kinasemodel6.apply(lambda x: 1 if x is not np.nan else 0)

    def generate_target_dict(self):
        def gen_dict(group):
            return {tid: act for tid, act in zip(group["AssayTarget"], group["active"])}

        group = self.df.groupby("Canonical_Smiles")
        temp = pd.DataFrame(group.apply(gen_dict))
        self.mt_df = pd.DataFrame(temp[0].tolist())
        self.mt_df["Canonical_Smiles"] = temp.index
        self.mt_df = self.mt_df.where((pd.notnull(self.mt_df)), -1)

    def generate_structures(self):
        structs = self.df[["Canonical_Smiles"]].drop_duplicates("Canonical_Smiles")
        structs["romol"] = structs.apply(lambda row: Chem.MolFromSmiles(row["Canonical_Smiles"]), axis=1)
        structs = structs.dropna()
        del structs["romol"]
        self.structs = structs

    def filter_and_save_data(self):
        self.mt_df = pd.merge(self.structs, self.mt_df, how="inner", on="Canonical_Smiles")
        self.mt_df = self.mt_df.replace(-1, 0)
        k = self.mt_df.sum()
        k.drop("Canonical_Smiles", inplace=True)
        k = k.astype(float)
        todrop = k[k < 15].index
        self.mt_df = self.mt_df.drop(todrop, axis=1)
        self.mt_df.to_csv(self.output_file, index=False, compression="gzip")

    def process_data(self):
        self.read_data()
        self.add_active_column()
        self.generate_target_dict()
        self.generate_structures()
        self.filter_and_save_data()


processor = DataProcessor(input_file, output_file)
processor.process_data()
