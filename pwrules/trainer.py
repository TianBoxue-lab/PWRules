import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from pathlib import Path
import d2l.torch
import os
from pwrules.utils.utils import load_config
from pwrules.utils.trail import fix_seed
from pwrules.dataloader import PWRulesDataLoader
from pwrules.model import PWRules
from pwrules.utils.metrics import get_all_metrics
import warnings
warnings.filterwarnings("ignore")


class PWRulesTrainer(object):
    def __init__(self, config_path):
        self.config = argparse.Namespace(**load_config(config_path))
        self.devices = d2l.torch.try_all_gpus()
        self.device = self.devices[self.config.gpu_id]
        self.model_version = f'{self.config.model_name}'
        self.model_weight_dir = Path(__file__).parent.parent / 'weight'
        if not os.path.exists(self.model_weight_dir):
            os.makedirs(self.model_weight_dir)

    def train(self, train_data_path, val_data_path):
        fix_seed(self.config.seed)

        dl_train, dl_valid = PWRulesDataLoader().get_dataloader(train_data_path, val_data_path, self.config.batch_size)

        model = PWRules(output_dim=self.config.output_dim)
        model.to(self.device)

        criterion = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.t_max)

        current_best_valid_score = -1
        current_best_epoch = 0
        for epoch in range(self.config.max_epoch):
            model.train()
            for x, y in dl_train:
                x, y = x.to(self.device), y.to(self.device).float()
                mask = ~torch.isnan(y)
                y = torch.where(mask, y, torch.zeros_like(y))
                pred = model(x)["predict"]
                loss = criterion(pred, y)
                loss = (loss * mask).sum() / mask.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            metric_train, metric_list_train = self.evaluate(model, dl_train)
            metric_valid, metric_list_valid = self.evaluate(model, dl_valid)

            score = metric_valid['auc']

            if score > current_best_valid_score:
                current_best_epoch = epoch
                current_best_valid_score = score
                torch.save(model.state_dict(), self.model_weight_dir / f'{self.model_version}.pkl')

            print("==================================================================================")
            print('Epoch', epoch)
            print('Train', metric_train)
            print('Valid', metric_valid)
            print('score', score)
            print('current_best_epoch', current_best_epoch, 'current_best_valid_score', current_best_valid_score)
            print("==================================================================================")

            if epoch > current_best_epoch + self.config.max_bearable_epoch or epoch == self.config.max_epoch - 1:
                print(f"{self.model_version} is Done!!")
                break

    def evaluate(self, model, data_loader, return_raw_data=False):
        model.eval()

        y_pred, y_true = [], []
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                pred = F.sigmoid(model(x)["predict"])
                y_pred.append(pred.cpu().numpy())
                y_true.append(y.numpy())
        y_pred, y_true = np.concatenate(y_pred, axis=0), np.concatenate(y_true, axis=0)

        metric, metric_list = get_all_metrics(y_true, y_pred)

        if return_raw_data:
            return metric, metric_list, y_true, y_pred
        else:
            return metric, metric_list

