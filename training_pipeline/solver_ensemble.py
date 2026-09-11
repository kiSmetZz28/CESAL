import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import logging
import datetime
import yaml

from utils.utils import *
from model.EMAT import EMAT
from data_factory.data_loader import get_loader_segment
from sklearn.mixture import GaussianMixture


def set_thresh_em(energy, n_components, covariance_type, max_iter, init_params, n_init):
    gm = GaussianMixture(
                 n_components=n_components,
                 covariance_type=covariance_type,
                 max_iter=max_iter,
                 init_params=init_params,
                 n_init=n_init,
                 random_state=42,
        ).fit(energy)
    pred = gm.predict(energy)
    return pred


def get_anomaly_ratio(em_pred):
    unique, counts = np.unique(em_pred, return_counts=True)
    total = len(em_pred)

    # Create a dictionary: label -> percentage
    label_percentages = {label: (count / total) * 100 for label, count in zip(unique, counts)}

    # Sort by percentage descending
    sorted_percentages = sorted(label_percentages.items(), key=lambda x: x[1], reverse=True)

    # Log counts and percentages
    logging.info(f"Label counts: {dict(zip(unique, counts))}")
    for label, percentage in sorted_percentages:
        logging.info(f"Label {label}: {percentage:.6f}%")

    return sorted_percentages


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logging.info('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path, hyperparameter):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path, hyperparameter)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path, hyperparameter)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path, hyperparameter):
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # num_epochs, k, e_layer_num, batch_size
        fileparam = f'e{hyperparameter[0]}_k{hyperparameter[1]}_l{hyperparameter[2]}_b{hyperparameter[3]}'
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_' + fileparam + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        ensemble_param = [self.num_epochs, self.k, self.e_layer_num, self.batch_size]

        self.train_loader = get_loader_segment(ensemble_param, self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(ensemble_param, self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(ensemble_param, self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(ensemble_param, self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def _update_threshold_config(self, thresh):
        """Update or append the EM threshold for this model into a YAML file.

        The target file depends on the dataset:
        - Openstack -> model_config/threshold_config/ensemble_config_os.yaml
        - BGL       -> model_config/threshold_config/ensemble_config_bgl.yaml
        - HDFS      -> model_config/threshold_config/ensemble_config_hdfs.yaml

        If the file does not exist, it will be created with a single model entry.
        If the model already exists, its threshold is updated; otherwise it is appended.
        """

        dataset_name = str(self.dataset).strip('"') if isinstance(self.dataset, str) else str(self.dataset)

        if dataset_name == 'Openstack':
            cfg_path = 'model_config/threshold_config/ensemble_config_os.yaml'
            model_prefix = 'Openstack_'
        elif dataset_name == 'BGL':
            cfg_path = 'model_config/threshold_config/ensemble_config_bgl.yaml'
            model_prefix = 'BGL_'
        elif dataset_name == 'HDFS':
            cfg_path = 'model_config/threshold_config/ensemble_config_hdfs.yaml'
            model_prefix = ''
        else:
            logging.warning(f"No threshold config mapping for dataset '{dataset_name}', skip writing threshold.")
            return

        # Build model name, consistent with checkpoint naming
        fileparam = f"e{self.num_epochs}_k{self.k}_l{self.e_layer_num}_b{self.batch_size}"
        model_name = f"{model_prefix}{fileparam}" if model_prefix else fileparam

        cfg_data = {}
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r') as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        cfg_data = loaded
            except Exception as exc:
                logging.warning(f"Failed to load existing threshold config '{cfg_path}': {exc}. Overwriting.")

        models = cfg_data.get('models', []) if isinstance(cfg_data.get('models', []), list) else []

        found = False
        for m in models:
            if m.get('name') == model_name:
                m['threshold'] = float(thresh)
                found = True
                break

        if not found:
            models.append({'name': model_name, 'threshold': float(thresh)})

        cfg_data['models'] = models

        try:
            with open(cfg_path, 'w') as f:
                yaml.safe_dump(cfg_data, f, sort_keys=False)
            logging.info(f"Saved threshold {thresh} for model '{model_name}' into '{cfg_path}'.")
        except Exception as exc:
            logging.warning(f"Failed to write threshold config '{cfg_path}': {exc}")

    def build_model(self):
        self.model = EMAT(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=self.e_layer_num)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        logging.info("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    logging.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            logging.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            logging.info(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            # num_epochs, k, e_layer_num, batch_size
            param = [self.num_epochs, self.k, self.e_layer_num, self.batch_size]
            early_stopping(vali_loss1, vali_loss2, self.model, path, param)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def singlemodelpred(self):
        # get the model checkpoint path
        fileparam = f'e{self.num_epochs}_k{self.k}_l{self.e_layer_num}_b{self.batch_size}'
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path),
                             str(self.dataset) + '_' + fileparam + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        logging.info(f"-----------------------Predicting model {fileparam}-----------------------")

        criterion = nn.MSELoss(reduction='none')

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # EM-GMM thresholding on the training set
        em_pred = set_thresh_em(train_energy.reshape(-1, 1), 7, 'tied', 100, 'k-means++', 10)
        train_pred = get_anomaly_ratio(em_pred)
        normal_ratio = train_pred[0][1]
        logging.info(f"Normal data ratio: {normal_ratio}")
        anomaly_ratio = 100 - normal_ratio
        logging.info(f"Abnormal data ratio: {anomaly_ratio}")

        # use train_energy to get the anomaly score, also need to use EM here.
        thresh = np.percentile(train_energy, normal_ratio) 
        logging.info(f"Threshold : {thresh}")

        # Save / update this threshold into the per-dataset threshold config YAML
        self._update_threshold_config(thresh)

        # Evaluation on the test set
        test_labels = []
        attens_energy = []

        for i, (input_data, labels) in enumerate(self.test_loader):

            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)


        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        logging.info(f"pred:   {pred.shape}")
        logging.info(f"gt:     {gt.shape}")

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        logging.info(f"pred:   {pred.shape}")
        logging.info(f"gt:     {gt.shape}")

        return pred, gt


    def test(self):
        logging.info("======================TEST MODE======================")

        pred, gt = self.singlemodelpred()

        logging.info(pred.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        logging.info(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))

        return accuracy, precision, recall, f_score
