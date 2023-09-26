from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

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
