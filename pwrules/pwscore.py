from pathlib import Path
from pwrules import RuleMatcher
from pprint import pprint
from tqdm import tqdm
import numpy as np
import pickle
from collections import defaultdict
from pwrules.utils.utils import check_path, load_pkl
from pwrules.fragment.frag_library import FragLibrary
from pwrules.utils.smiles import read_smiles_dict_from_smi, get_longest_molecule_component


class PWScore(object):
    def __init__(self, rule_dir):
        self.rule_dir = rule_dir
        self.frag_id_list = load_pkl(Path(__file__).parent.parent / 'data/frag_id_list.pkl')

    def screen_library(self, fasta_path, uniprot2word_path, library_path, top_k):
        matcher = RuleMatcher(self.rule_dir)
        uniprot2matched_result = matcher.predict_fragments(fasta_path, uniprot2word_path)

        if not check_path(library_path):
            raise ValueError(f'File not found: {library_path}')

        frag_library = FragLibrary()
        frag2frequency = get_frag2frequency(frag_library)
        smiles2frags = get_smiles2frag_atom(library_path, frag_library)

        uniprot_list = sorted(list(uniprot2matched_result.keys()))
        return _drug_screen(uniprot_list, uniprot2matched_result, smiles2frags, frag2frequency, top_k)


def _drug_screen(uniprot_list, uniprot2matched_result, smiles2frags, frag2freq, top_k):
    result = {}
    for index, uniprot_id in enumerate(uniprot_list):
        privileged_frag2score = uniprot2matched_result[uniprot_id]['Privileged_frag']
        privileged_frag2score = {k: v * frag2freq[k] for k, v in privileged_frag2score.items()}
        sorted_privileged_frag2score = sorted(privileged_frag2score.items(), key=lambda x: x[1], reverse=True)

        smiles2score = {}
        for smi in smiles2frags:
            score = get_pwscore(smiles2frags[smi], sorted_privileged_frag2score)
            smiles2score[smi] = score
        result[uniprot_id] = sorted(smiles2score.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return result


def get_pwscore(frag2atoms, privileged_frag2score, n=10):
    _pwscore = 0
    matched_atom = defaultdict(int)
    for privileged_frag, score in privileged_frag2score:
        if privileged_frag not in frag2atoms:
            continue
        atoms = frag2atoms[privileged_frag]

        if not is_good_frag(matched_atom, atoms, n=n):
            continue

        for atom in atoms:
            matched_atom[atom] += 1
        _pwscore += score
    return _pwscore


def is_good_frag(matched_atom, atoms, n=1):
    for atom in atoms:
        if atom in matched_atom and matched_atom[atom] >= n:
            return False
    return True


def get_frag2frequency(frag_library):
    frag2frequency = {}
    for k, v in frag_library.frag_library.items():
        frequency = v['Frequency']
        frag2frequency[k] = np.log10(frequency)
    min_, max_ = min(frag2frequency.values()), max(frag2frequency.values())
    frag2frequency = {k: 1 - (v - min_) / (max_ - min_) for k, v in frag2frequency.items()}
    return frag2frequency


def get_smiles2frag_atom(library_path, frag_library):
    smiles2frag_path = f'{library_path[:-4]}_smi2frag_atom.pkl'
    if check_path(smiles2frag_path):
        return load_pkl(smiles2frag_path)

    smiles_list = list(read_smiles_dict_from_smi(library_path).keys())
    result = {}
    for index, smi in tqdm(enumerate(smiles_list), total=len(smiles_list)):
        matched_frag = frag_library.match_frag_atom(get_longest_molecule_component(smi, use_atom_count=False))
        if matched_frag is None:
            continue
        result[smi] = matched_frag

    with open(smiles2frag_path, 'wb') as f:
        pickle.dump(result, f)
    return result
