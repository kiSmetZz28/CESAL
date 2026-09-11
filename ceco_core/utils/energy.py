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
    """Compute attention-based anomaly energy scores for a batch of windows.

    Mirrors the scoring logic in Solver.singlemodelpred. The model must
    already be in eval mode when this function is called.

    Parameters
    ----------
    model : nn.Module
        EM-AT model (eval mode, on target device).
    x : torch.Tensor
        Input batch of shape [B, win_size, n_features].
    win_size : int
        Window length (must match model's win_size).
    temperature : float
        Scaling factor for KL loss terms (default: 50).

    Returns
    -------
    np.ndarray
        1-D array of energy values, length B (one score per window).
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
