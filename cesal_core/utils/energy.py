import numpy as np
import torch
import torch.nn as nn


def my_kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence term used in the Anomaly Transformer loss."""
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def compute_energy_batch(
    model: nn.Module,
    x: torch.Tensor,
    win_size: int,
    temperature: float = 50,
) -> np.ndarray:
    """Attention-based anomaly energy for a batch [B, win_size, n_features], scored as in
    Solver.singlemodelpred (model already in eval mode; ``temperature`` scales the KL terms).

    Returns [B, win_size]: one score per time step. Flatten with reshape(-1) for per-line
    scores (lad_bat_cloud.py) or average for one score per window (dashboard/bat_predict.py).
    """
    criterion = nn.MSELoss(reduction='none')
    device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        output, series, prior, _ = model(x)
        loss = torch.mean(criterion(x, output), dim=-1)

        series_loss = prior_loss = 0.0
        for u in range(len(prior)):
            norm = (
                prior[u]
                / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                .repeat(1, 1, 1, win_size)
            )
            kl_s = my_kl_loss(series[u], norm.detach()) * temperature
            kl_p = my_kl_loss(norm, series[u].detach()) * temperature
            if u == 0:
                series_loss, prior_loss = kl_s, kl_p
            else:
                series_loss += kl_s
                prior_loss += kl_p

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        energy = (metric * loss).detach().cpu().numpy()

    return energy
