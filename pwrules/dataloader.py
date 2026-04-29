from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from pwrules.utils.utils import load_pkl


class PWRulesDataset(Dataset):
    def __init__(self, input_data):
        self.input_data = input_data
        self.uniprot_ids = sorted(input_data.keys())

    def __len__(self):
        return len(self.uniprot_ids)

    def __getitem__(self, index):
        x = self.input_data[self.uniprot_ids[index]]['Words_embed']
        y = self.input_data[self.uniprot_ids[index]]['Label']
        y = torch.from_numpy(y)
        return x, y


class PWRulesDataLoader(object):
    def __init__(self):
        pass

    def get_dataloader(self, train_path, val_path, batch_size):
        train, valid = self.get_dataset(train_path, val_path)

        train = PWRulesDataset(train)
        valid = PWRulesDataset(valid)

        train = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid = DataLoader(dataset=valid, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return train, valid

    def get_dataset(self, train_path, val_path):
        train = load_pkl(train_path)
        valid = load_pkl(val_path)
        return train, valid


def collate_fn(batch):
    x_list, y_list = zip(*batch)
    padded_x = pad_sequence(x_list, batch_first=True, padding_value=0)
    y = torch.stack(y_list)
    return padded_x, y
