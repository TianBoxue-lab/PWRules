from pathlib import Path
from pwrules.utils.utils import load_pkl
from rdkit import Chem
import numpy as np


class FragLibrary(object):
    def __init__(self):
        self.frag_library = load_pkl(Path(__file__).parent.parent / f"data/frag_library.pkl")

    def match_frag_atom(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        match_smarts_id = {}
        for frag_id, pattern in self.frag_library.items():
            matches = mol.GetSubstructMatches(pattern['Pattern'])
            if len(matches) > 0:
                match_smarts_id[frag_id] = np.array(list(set().union(*matches)), dtype=np.uint8)
        return match_smarts_id
