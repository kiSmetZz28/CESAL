import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.mixture import GaussianMixture

from ceco_core.data.loaders import get_loader_segment
from ceco_core.models.EMAT import EMAT
from ceco_core.utils.energy import my_kl_loss
from ceco_core.utils.metrics import evaluate as _evaluate


def _fit_gmm(energy: np.ndarray, n_components: int, covariance_type: str,
             max_iter: int, init_params: str, n_init: int) -> np.ndarray:
    gm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        init_params=init_params,
        n_init=n_init,
        random_state=42,
    ).fit(energy)
    return gm.predict(energy)


def _log_cluster_percentages(em_pred: np.ndarray):
    unique, counts = np.unique(em_pred, return_counts=True)
    total = len(em_pred)
    label_pct = {label: (count / total) * 100 for label, count in zip(unique, counts)}
    sorted_pct = sorted(label_pct.items(), key=lambda x: x[1], reverse=True)
    logging.info("Label counts: %s", dict(zip(unique, counts)))
    for label, pct in sorted_pct:
        logging.info("  Label %d: %.6f%%", label, pct)
    return sorted_pct


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, lr: float) -> None:
    new_lr = lr * (0.5 ** ((epoch - 1) // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    logging.info("Learning rate updated to %g", new_lr)


class EarlyStopping:
    def __init__(self, patience: int = 7, verbose: bool = False,
                 dataset_name: str = '', delta: float = 0):
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
            self._save(val_loss, val_loss2, model, path, hyperparameter)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            logging.info("EarlyStopping counter: %d / %d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self._save(val_loss, val_loss2, model, path, hyperparameter)
            self.counter = 0

    def _save(self, val_loss, val_loss2, model, path, hyperparameter):
        if self.verbose:
            logging.info(
                "Validation loss decreased (%.6f -> %.6f). Saving model...",
                self.val_loss_min, val_loss,
            )
        e, k, l, b = hyperparameter
        filename = f"{self.dataset}_e{e}_k{k}_l{l}_b{b}_checkpoint.pth"
        torch.save(model.state_dict(), os.path.join(path, filename))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


# Maps dataset name → (threshold output path, model name prefix)
_THRESHOLD_OUTPUT = {
    'Openstack': ('outputs/os/thresholds_cloud.yaml',   'Openstack_'),
    'BGL':       ('outputs/bgl/thresholds_cloud.yaml',  'BGL_'),
    'HDFS':      ('outputs/hdfs/thresholds_cloud.yaml', 'HDFS_'),
}


class Solver:
    DEFAULTS: dict = {}

    def __init__(self, config: dict):
        self.__dict__.update(Solver.DEFAULTS, **config)

        ensemble_param = [self.num_epochs, self.k, self.e_layer_num, self.batch_size]

        self.train_loader = get_loader_segment(
            ensemble_param, self.data_path,
            batch_size=self.batch_size, win_size=self.win_size,
            mode='train', dataset=self.dataset,
        )
        self.vali_loader = get_loader_segment(
            ensemble_param, self.data_path,
            batch_size=self.batch_size, win_size=self.win_size,
            mode='val', dataset=self.dataset,
        )
        self.test_loader = get_loader_segment(
            ensemble_param, self.data_path,
            batch_size=self.batch_size, win_size=self.win_size,
            step=self.win_size, mode='test', dataset=self.dataset,
        )
        self.thre_loader = get_loader_segment(
            ensemble_param, self.data_path,
            batch_size=self.batch_size, win_size=self.win_size,
            mode='thre', dataset=self.dataset,
        )

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self) -> None:
        self.model = EMAT(
            win_size=self.win_size,
            enc_in=self.input_c,
            c_out=self.output_c,
            e_layers=self.e_layer_num,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader) -> tuple:
        self.model.eval()
        loss_1, loss_2 = [], []

        for input_data, _ in vali_loader:
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            series_loss = prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (
                    torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                        .repeat(1, 1, 1, self.win_size)).detach()))
                    + torch.mean(my_kl_loss((
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                        .repeat(1, 1, 1, self.win_size)).detach(), series[u]))
                )
                prior_loss += (
                    torch.mean(my_kl_loss((
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                        .repeat(1, 1, 1, self.win_size)), series[u].detach()))
                    + torch.mean(my_kl_loss(series[u].detach(), (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                        .repeat(1, 1, 1, self.win_size))))
                )
            series_loss /= len(prior)
            prior_loss /= len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self) -> None:
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

                series_loss = prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (
                        torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                            .repeat(1, 1, 1, self.win_size)).detach()))
                        + torch.mean(my_kl_loss((
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                            .repeat(1, 1, 1, self.win_size)).detach(), series[u]))
                    )
                    prior_loss += (
                        torch.mean(my_kl_loss((
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                            .repeat(1, 1, 1, self.win_size)), series[u].detach()))
                        + torch.mean(my_kl_loss(series[u].detach(), (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                            .repeat(1, 1, 1, self.win_size))))
                    )
                series_loss /= len(prior)
                prior_loss /= len(prior)

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

                # Minimax: two backward passes to optimize series vs prior divergence
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            logging.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)
            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            logging.info(
                "Epoch %d, Steps %d | Train Loss: %.7f  Vali Loss: %.7f",
                epoch + 1, train_steps, train_loss, vali_loss1,
            )

            param = [self.num_epochs, self.k, self.e_layer_num, self.batch_size]
            early_stopping(vali_loss1, vali_loss2, self.model, path, param)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def singlemodelpred(self) -> tuple:
        fileparam = f"e{self.num_epochs}_k{self.k}_l{self.e_layer_num}_b{self.batch_size}"
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path),
                             str(self.dataset) + '_' + fileparam + '_checkpoint.pth')
            )
        )
        self.model.eval()
        temperature = 50

        logging.info("-----------------------Predicting model %s-----------------------", fileparam)

        criterion = nn.MSELoss(reduction='none')

        # (1) Compute energy on train set
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

        # (2) Compute energy on thre set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
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
        thre_energy = np.array(attens_energy)

        # Combine train + thre energy for EM-GMM threshold selection
        # combined_energy = np.concatenate([train_energy, thre_energy])
        combined_energy = train_energy

        em_pred = _fit_gmm(combined_energy.reshape(-1, 1), 7, 'tied', 100, 'k-means++', 10)
        sorted_pct = _log_cluster_percentages(em_pred)
        normal_ratio = sorted_pct[0][1]
        logging.info("Normal data ratio: %s", normal_ratio)
        logging.info("Abnormal data ratio: %s", 100 - normal_ratio)

        thresh = float(np.percentile(combined_energy, normal_ratio))
        logging.info("Threshold: %g", thresh)
        self._update_threshold_config(thresh)

        # (3) Compute energy on the test set
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

        logging.info("pred:   %s", pred.shape)
        logging.info("gt:     %s", gt.shape)

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
        logging.info("pred:   %s", pred.shape)
        logging.info("gt:     %s", gt.shape)

        return pred, gt

    def test(self) -> None:
        logging.info("======================TEST MODE======================")
        pred, gt = self.singlemodelpred()
        _evaluate(gt, pred)

    def _update_threshold_config(self, thresh: float) -> None:
        """Write or update the per-model EM-GMM threshold in the dataset YAML."""
        dataset_name = str(self.dataset).strip('"')

        if dataset_name not in _THRESHOLD_OUTPUT:
            logging.warning("No threshold config mapping for dataset '%s', skipping.", dataset_name)
            return

        cfg_path, prefix = _THRESHOLD_OUTPUT[dataset_name]
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)

        fileparam = f"e{self.num_epochs}_k{self.k}_l{self.e_layer_num}_b{self.batch_size}"
        model_name = f"{prefix}{fileparam}" if prefix else fileparam

        cfg_data: dict = {}
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r') as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        cfg_data = loaded
            except Exception as exc:
                logging.warning("Failed to load '%s': %s. Overwriting.", cfg_path, exc)

        models = cfg_data.get('models', [])
        if not isinstance(models, list):
            models = []

        for m in models:
            if m.get('name') == model_name:
                m['threshold'] = float(thresh)
                break
        else:
            models.append({'name': model_name, 'threshold': float(thresh)})

        cfg_data['models'] = models

        try:
            with open(cfg_path, 'w') as f:
                yaml.safe_dump(cfg_data, f, sort_keys=False)
            logging.info("Saved threshold %g for model '%s' into '%s'.", thresh, model_name, cfg_path)
        except Exception as exc:
            logging.warning("Failed to write threshold config '%s': %s", cfg_path, exc)
