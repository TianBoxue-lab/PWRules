from pathlib import Path
from collections import defaultdict
from pwrules.utils.utils import check_path, load_pkl
from pwrules.utils.sequence import read_fasta


class RuleMatcher(object):
    def __init__(self, rule_dir):
        self.rule_dir = rule_dir
        self.frag_id_list = load_pkl(Path(__file__).parent.parent / 'data/frag_id_list.pkl')

    def predict_fragments(self, fasta_path, uniprot2word_path):
        if not check_path(fasta_path):
            raise ValueError(f'File not found: {fasta_path}')

        uniprot2seq = read_fasta(fasta_path)
        uniprot2words = get_protein_words(uniprot2seq, uniprot2word_path)

        return self.match_pair_rules(uniprot2words)

    def match_pair_rules(self, uniprot2words):
        word2frags = load_pkl(f'{self.rule_dir}/word2frags.pkl')
        word_frag2score = load_pkl(f'{self.rule_dir}/word_frag2score.pkl')
        best_threshold = load_pkl(f'{self.rule_dir}/best_thresholds.pkl')

        result = {}
        for uniprot, data in uniprot2words.items():
            frag2score = defaultdict(list)
            for word in data['Words']:
                frags = word2frags.get(word, {})
                for frag in frags:
                    frag2score[frag].append(word_frag2score[f'{word} {frag}'])

            privileged_frag = {}
            for frag, threshold in zip(self.frag_id_list, best_threshold):
                if frag not in frag2score:
                    continue
                non_prob = 1
                for i in frag2score[frag]:
                    non_prob *= (1 - i)
                score = 1 - non_prob
                if score > threshold:
                    privileged_frag[frag] = score

            matched_word2frags = defaultdict(set)
            matched_functional_words = set()
            for word in data['Words']:
                frags = word2frags.get(word, {})
                for frag in frags:
                    if frag in privileged_frag:
                        matched_word2frags[word].add(frag)
                        matched_functional_words.add(word)

            result[uniprot] = {
                'Privileged_frag': privileged_frag,
                'Matched_pairs': matched_word2frags,
                'Matched_functional_words': matched_functional_words,
            }

        return result


def get_protein_words(uniprot2seq, ww_path):
    uniprot2words = {}
    ww = load_pkl(ww_path)
    for uniprot_id, sequence in uniprot2seq.items():
        if uniprot_id not in ww:
            continue

        uniprot2words[uniprot_id] = {
            'Sequence': sequence,
            'Words': ww[uniprot_id]['Words']
        }
    return uniprot2words
