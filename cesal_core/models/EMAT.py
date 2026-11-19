"""EM-AT base learner (paper Sec. 3.5.1): the Anomaly Transformer encoder plus
EM-GMM automated thresholding.
"""
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding


# EM-GMM automated thresholding
GMM_RANDOM_STATE = 42
# The fitted mixture can depend on the BLAS/OpenMP thread count (HDFS's ~11M energies),
# so it is fitted on one thread: the released thresholds were, and the result is then
# the same on any machine.
GMM_THREADS = 1


def fit_gmm(
    energy: np.ndarray,
    n_components: int = 7,
    covariance_type: str = 'tied',
    max_iter: int = 100,
    init_params: str = 'k-means++',
    n_init: int = 10,
) -> np.ndarray:
    """Fit a Gaussian mixture with EM to energies of shape [N, 1]; return each one's component."""
    from sklearn.mixture import GaussianMixture
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=GMM_THREADS):
        gm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            init_params=init_params,
            n_init=n_init,
            random_state=GMM_RANDOM_STATE,
        ).fit(energy)
        return gm.predict(energy)


def cluster_percentages(labels: np.ndarray) -> List[Tuple[int, float]]:
    """Share of the points (%) in each mixture component, largest first."""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    label_pct = {label: (count / total) * 100 for label, count in zip(unique, counts)}
    sorted_pct = sorted(label_pct.items(), key=lambda x: x[1], reverse=True)
    logging.debug("Label counts: %s", dict(zip(unique, counts)))
    for label, pct in sorted_pct:
        logging.debug("  Label %d: %.6f%%", label, pct)
    return sorted_pct


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class EMAT(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(EMAT, self).__init__()
        self.output_attention = output_attention

        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out

    @staticmethod
    def fit_threshold(
        energy: np.ndarray,
        n_components: int = 7,
        covariance_type: str = 'tied',
        max_iter: int = 100,
        init_params: str = 'k-means++',
        n_init: int = 10,
    ) -> Tuple[float, float, List[Tuple[int, float]]]:
        """EM-GMM automated thresholding: fit a Gaussian mixture to the energies with EM,
        take the largest component's share as the normal ratio (%), and set λ to the
        energy at that percentile. Returns (λ, normal_ratio, every component's share).
        An event is anomalous when its energy is strictly greater than λ.
        """
        energy = np.asarray(energy)
        labels = fit_gmm(energy.reshape(-1, 1), n_components, covariance_type,
                         max_iter, init_params, n_init)
        cluster_pcts = cluster_percentages(labels)
        normal_ratio = cluster_pcts[0][1]
        threshold = float(np.percentile(energy, normal_ratio))
        return threshold, normal_ratio, cluster_pcts
