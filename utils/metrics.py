import numpy as np
from tqdm import tqdm


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def PMSE(pred, true):
    pred = pred[..., 0]
    true = true[..., 0]
    left = true[:, :-2]
    middle = true[:, 1:-1]
    right = true[:, 2:]
    mean = true.mean(axis=1, keepdims=True)
    peaks = (middle > left) & (middle > right) & (middle > mean)
    mse = np.mean(np.square(true[:, 1:-1][peaks] - pred[:, 1:-1][peaks]))
    return mse


def PMAE(pred, true):
    pred = pred[..., 0]
    true = true[..., 0]
    left = true[:, :-2]
    middle = true[:, 1:-1]
    right = true[:, 2:]
    mean = true.mean(axis=1, keepdims=True)
    peaks = (middle > left) & (middle > right) & (middle > mean)
    mae = np.mean(np.abs(true[:, 1:-1][peaks] - pred[:, 1:-1][peaks]))
    return mae


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    pmse = PMSE(pred, true)
    pmae = PMAE(pred, true)

    return mae, mse, rmse, mape, mspe, pmse, pmae

import numpy as np

def peaks_mask(x: np.ndarray, n: int = 6):
    """
    Returns a boolean array of the same shape as x
    """
    middle = x[..., n:-n]
    mean = np.mean(x, axis=-1, keepdims=True)
    s = x.shape[-1]
    peaks = np.all([middle >= x[..., n - i:s - n - i] for i in range(1, n + 1)], axis=0) & \
            np.all([middle >= x[..., n + i:s - n + i] for i in range(1, n + 1)], axis=0) & \
            (middle > mean)

    # extend to the same shape as x
    zeros = np.zeros_like(x[..., 0:n], dtype=bool)
    peaks = np.concatenate([zeros, peaks, zeros], axis=-1)
    return peaks

def pnorm(y_pred: np.ndarray, y_true: np.ndarray, p: int = 2):
    """
    p = 1: MAE
    p = 2: MSE
    """
    peaks = peaks_mask(y_true)
    if p == 1:
        norm = np.mean(np.abs(y_true[peaks] - y_pred[peaks]), axis=-1)
    elif p == 2:
        norm = np.mean((y_true[peaks] - y_pred[peaks]) ** 2, axis=-1)
    else:
        raise ValueError("p must be 1 or 2")
    return norm

def p3sw(y_pred: np.ndarray, y_true: np.ndarray, T: int = 3):
    """
    Implements the following formula:
    P3_{sw}(y, \hat{y}) = \sum_{t \in P} \big(y(t) - \max_{t' \in [t - T, t + T]} \hat{y}(t') \big)^2
    where P is the set of peaks in y
    """
    peaks = peaks_mask(y_true)
    y_true_peaks = y_true[peaks]

    # Calculate max(y_pred) in range [t - T, t + T]
    y_pred_padded = np.pad(y_pred, pad_width=((0, 0), (T, T)))
    y_pred_windows = np.stack([y_pred_padded[:, i:i + 2 * T + 1] for i in range(y_true.shape[1])], axis=1)
    y_pred_max_windows = np.max(y_pred_windows, axis=-1)

    pshift1 = np.mean((y_true_peaks - y_pred_max_windows[peaks]) ** 2, axis=-1)
    return pshift1

def p3eu(y_pred: np.ndarray, y_true: np.ndarray, alpha: float = 0.2, beta: float = 1.0, T: int = 10):
    peaks = peaks_mask(y_true)
    indices = np.arange(y_true.shape[1]) * np.sqrt(alpha)
    indices = np.vstack([indices] * y_true.shape[0])
    y_true *= np.sqrt(beta)
    y_pred *= np.sqrt(beta)
    y_true = np.stack([y_true, indices], axis=-1)
    y_pred = np.stack([y_pred, indices], axis=-1)
    y_true_peaks = y_true[peaks]

    y_pred_padded = np.pad(y_pred, pad_width=((0, 0), (T, T), (0, 0)))
    y_pred_windows = np.stack([y_pred_padded[:, i:i + 2 * T + 1] for i in range(y_true.shape[1])], axis=1)
    y_pred_peak_windows = y_pred_windows[peaks]
    distances = np.sum((y_pred_peak_windows - y_true_peaks[:, None]) ** 2, axis=-1)
    distances = np.min(distances, axis=-1)
    pshift2 = np.mean(distances, axis=-1)
    return pshift2

def all_peak_metrics(yp: np.ndarray, yt: np.ndarray):
    """
    yp.shape = yt.shape = (n_samples, n_timesteps)
    """
    metrics = {
        "mse": MSE,
        "mae": MAE,
        "pmse": lambda yt, yp: (pnorm(yt, yp, p=2) + pnorm(yp, yt, p=2))/2,
        "pmae": lambda yt, yp: (pnorm(yt, yp, p=1) + pnorm(yp, yt, p=1))/2,
        "p3sw": lambda yt, yp: (p3sw(yt, yp) + p3sw(yp, yt))/2,
        "p3eu": lambda yt, yp: (p3eu(yt, yp) + p3eu(yp, yt))/2,
    }
    return {k: v(yt, yp) for k, v in metrics.items()}

"""
Using Pytorch:

def peaks_mask(x: torch.Tensor):
    left = x[..., :-2]
    middle = x[..., 1:-1]
    right = x[..., 2:]
    mean = torch.mean(x, dim=-1, keepdim=True)
    peaks = (middle > left) & (middle > right) & (middle > mean)
    # extend to the same shape as x
    zeros = torch.zeros_like(x[..., 0:1], dtype=torch.bool)
    peaks = torch.cat([zeros, peaks, zeros], dim=-1)
    return peaks


def pnorm(y_pred: torch.Tensor, y_true: torch.Tensor, p: int = 2):
    peaks = peaks_mask(y_true)
    norm = torch.norm(y_true[peaks] - y_pred[peaks], p=p, dim=-1)
    return norm


def p3sw(y_pred: torch.Tensor, y_true: torch.Tensor, T: int = 10):
    peaks = peaks_mask(y_true)
    y_true_peaks = y_true[peaks]

    # Calculate max(y_pred) in range [t - T, t + T]
    y_pred_padded = torch.pad(y_pred, pad=(T, T))
    y_pred_windows = torch.stack([y_pred_padded[:, i:i + 2 * T + 1] for i in range(y_true.shape[1])], dim=1)
    y_pred_max_windows = torch.max(y_pred_windows, dim=-1).values

    # Calculate PSHIFT1
    pshift1 = torch.mean((y_true_peaks - y_pred_max_windows[peaks]) ** 2, dim=-1)

    return pshift1


def p3eu(y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float = 1.0, beta: float = 1.0):
    y_true_peaks = peaks_mask(y_true)
    y_pred_peaks = peaks_mask(y_pred)

    true_peak_indices = torch.where(y_true_peaks)
    pred_peak_indices = torch.where(y_pred_peaks)

    # Compute minimum distance between peaks
    min_distances = []
    for true_peak_idx in true_peak_indices:
        distances = []
        for pred_peak_idx in pred_peak_indices:
            time_diff = (torch.tensor(float(true_peak_idx[1] - pred_peak_idx[1])))**2
            y_diff = (y_true[true_peak_idx[0], true_peak_idx[1]] - y_pred[pred_peak_idx[0], pred_peak_idx[1]])**2
            distance = alpha * time_diff + beta * y_diff
            distances.append(distance)

        min_distances.append(torch.min(torch.tensor(distances)))

    pshift2 = torch.mean(torch.tensor(min_distances))

    return pshift2
"""