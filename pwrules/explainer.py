import numpy as np
import torch
import argparse
import pickle
from collections import defaultdict
from pwrules.utils.trail import fix_seed
from pwrules.utils.utils import load_pkl, load_config
from pwrules.dataloader import PWRulesDataLoader
from pwrules.model import PWRules
from pathlib import Path
from captum.attr import IntegratedGradients
import os
import warnings
warnings.filterwarnings('ignore')


class RuleExtractor(object):
    def __init__(self, config_path, checkpoint_path):
        self.config = argparse.Namespace(**load_config(config_path))
        self.checkpoint_path = checkpoint_path
        self.device = 'cpu'
        self.model_version = f'{self.config.model_name}'
        self.model_weight_dir = Path(__file__).parent.parent / 'weight'
        self.output_dir = Path(__file__).parent.parent / 'rules'
        self.frag_id_list = load_pkl(Path(__file__).parent.parent / 'data/frag_id_list.pkl')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_rules(self, train_data_path, val_data_path):
        fix_seed(self.config.seed)

        train, valid = PWRulesDataLoader().get_dataset(train_data_path, val_data_path)

        train, valid = self.get_explain_data(train), self.get_explain_data(valid)
        dataset_dict = {
            'train': train,
            'valid': valid,
        }
        self.explain(dataset_dict)

    def explain(self, dataset_dict):
        model = PWRules(output_dim=self.config.output_dim)
        model.load_state_dict(torch.load(self.model_weight_dir / f'{self.model_version}.pkl', map_location=self.device))
        model.to(self.device)
        model.eval()

        def forward_func(x):
            padding_mask = (x.abs().sum(dim=-1) == 0)
            x = x.transpose(0, 1)
            for layer_idx, layer in enumerate(model.layers):
                x, attn = layer(x, self_attn_padding_mask=padding_mask, need_head_weights=False)
            x = model.emb_layer_norm_after(x)
            x = x.transpose(0, 1)
            x = model.mlp(x[:, 0, :])
            return x

        lig = IntegratedGradients(forward_func)

        word2frags = defaultdict(set)
        word_frag2score = defaultdict(list)
        for dataset_name, dataset in dataset_dict.items():
            print(f'==={dataset_name}===')

            n = len(dataset)
            for index, data in enumerate(dataset):
                if index % 5 == 0:
                    print(f'{index} / {n}')
                words = data['Words']
                word_poses = data['Word_poses']
                words_embed = data['Words_embed']
                words_num = len(words)
                label = data['Label']

                words_embed = add_cls_embedding(model, words_embed.unsqueeze(0))
                baseline = torch.zeros((1, words_num, 1280), dtype=torch.float32)
                baseline = add_cls_embedding(model, baseline)

                with torch.no_grad():
                    output = torch.sigmoid(forward_func(words_embed)).flatten().numpy()

                for label_index, label_name in enumerate(self.frag_id_list):
                    if label[label_index] <= 0.5 or np.isnan(label[label_index]):
                        continue
                    pred = output[label_index]
                    if pred <= 0.5:
                        continue

                    attributions = lig.attribute(target=label_index, inputs=words_embed, baselines=baseline)
                    attributions = attributions.sum(dim=-1).squeeze(0)
                    attributions = (attributions / torch.norm(attributions)).numpy()
                    total_attribution = np.sum(attributions[attributions > 0])

                    important_words = []
                    for seq_word_index in range(words_num):
                        attri = attributions[seq_word_index + 1]
                        if attri > 0:
                            important_words.append([words[seq_word_index], word_poses[seq_word_index], attri])
                    important_words = sorted(important_words, key=lambda x: x[-1], reverse=True)

                    top_word_pos = set()
                    alpha = 0.5
                    sum_attribution = 0
                    for i in important_words:
                        if sum_attribution > alpha * total_attribution:
                            break
                        sum_attribution += i[2]
                        top_word_pos = top_word_pos | set(i[1])

                        word2frags[i[0]].add(label_name)

                        rule_score = (pred * i[2]) ** 0.5
                        word_frag2score[f'{i[0]} {label_name}'].append(rule_score)

        word_frag2score = {k: np.mean(v) for k, v in word_frag2score.items()}

        with open(f'{self.output_dir}/word2frags.pkl', 'wb') as f:
            pickle.dump(word2frags, f)
        with open(f'{self.output_dir}/word_frag2score.pkl', 'wb') as f:
            pickle.dump(word_frag2score, f)

    def get_explain_data(self, data_dict):
        dataset = sorted([{**v, 'Uniprot id': k} for k, v in data_dict.items()], key=lambda x: x['Uniprot id'])
        return dataset


def add_cls_embedding(model, x):
    cls_input = torch.zeros((x.shape[0],), dtype=torch.long).to(x.device)
    cls_embedding = model.embedding_layer(cls_input).unsqueeze(1)
    x = torch.cat((cls_embedding, x), dim=1).detach()
    return x

