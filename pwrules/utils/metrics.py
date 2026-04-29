import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score
from pwrules.utils.utils import nan_mean
from tqdm import tqdm


def mcc_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    if denominator == 0:
        return np.nan
    return numerator / denominator


def get_all_metrics(y_true, y_pred, binary_threshold=0.5):
    y_pred_binary = np.where(y_pred >= binary_threshold, 1, 0)
    y_true_binary = np.where(np.isnan(y_true), y_true, (y_true >= binary_threshold).astype(int))

    metric_list = {
        'auc': [],
        'ap': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'mcc': []
    }
    for i in tqdm(range(y_true.shape[1]), desc='get metrics'):
        col_true, col_pred = y_true[:, i], y_pred[:, i]
        col_true_binary, col_pred_binary = y_true_binary[:, i], y_pred_binary[:, i]
        mask = ~np.isnan(col_true)
        col_true, col_pred = col_true[mask], col_pred[mask]
        col_true_binary, col_pred_binary = col_true_binary[mask], col_pred_binary[mask]
        mask = ~np.isnan(col_pred)
        col_true, col_pred = col_true[mask], col_pred[mask]
        col_true_binary, col_pred_binary = col_true_binary[mask], col_pred_binary[mask]
        if len(col_true) == 0:
            for key in metric_list.keys():
                metric_list[key].append(np.nan)
            continue

        try:
            metric_list['auc'].append(roc_auc_score(col_true_binary, col_pred))
        except:
            metric_list['auc'].append(np.nan)
        try:
            metric_list['ap'].append(average_precision_score(col_true_binary, col_pred))
        except:
            metric_list['ap'].append(np.nan)

        metric_list['accuracy'].append(accuracy_score(col_true_binary, col_pred_binary))
        metric_list['precision'].append(precision_score(col_true_binary, col_pred_binary, zero_division=np.nan))
        metric_list['recall'].append(recall_score(col_true_binary, col_pred_binary, zero_division=np.nan))
        metric_list['f1'].append(f1_score(col_true_binary, col_pred_binary, zero_division=np.nan))
        metric_list['mcc'].append(mcc_score(col_true_binary, col_pred_binary))

    metric = {k: round(nan_mean(v), 3) for k, v in metric_list.items()}
    return metric, metric_list
