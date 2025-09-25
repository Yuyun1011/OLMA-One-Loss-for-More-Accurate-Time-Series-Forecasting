"""
OLMA can replace the original loss function of any supervised 
time series forecasting model. In most open-source implement-
ations, simply replace 'loss = criterion(outputs, batch_y)'
with 'loss = OLMA(outputs, batch_y)' inside the train function
in /exp/exp_main.py to use OLMA.
"""

import torch
import torch.nn.functional as F
import numpy as np

def Wavelet_Transformer(x, wavelet='haar'):
    """
    Wavelet transform on input [B, L, C].
    Compute both low-frequency (approximation) and high-frequency (detail),
    then concatenate along the time dimension.

    Args:
        x: Tensor of shape [B, L, C]

    Returns:
        Tensor of shape [B, L_concat, C],
        where L_concat = 2 * ceil(L / 2).
    """
    B, L, C = x.shape
    need_pad = (L % 2 != 0)
    if need_pad:
        # Pad one value at the end of the time dimension if L is odd
        x = F.pad(x, (0, 0, 0, 1))

    # Change to [B, C, L] for depthwise conv
    x_bc = x.permute(0, 2, 1)

    # choose wavelet: "haar", "db2", "sym2", "db4", "coif1", or "bior1.3"
    device = x.device

    if wavelet == "haar":
        # Haar low-pass
        s = 1 / np.sqrt(2)
        dec_lo_np = np.array([s, s], dtype=np.float32)

    elif wavelet == "db2" or wavelet == "sym2":
        # Daubechies 2 / Symlet 2 low-pass (same coefficients for order=2)
        dec_lo_np = np.array([
            -0.1294095226,  0.2241438680,
            0.8365163037,  0.4829629131
        ], dtype=np.float32)

    elif wavelet == "db4":
        # Daubechies 4 low-pass
        dec_lo_np = np.array([
            -0.0105974018,  0.0328830117,  0.0308413818, -0.1870348117,
            -0.0279837694,  0.6308807679,  0.7148465706,  0.2303778133
        ], dtype=np.float32)

    elif wavelet == "coif1":
        # Coiflet 1 low-pass
        dec_lo_np = np.array([
            -0.0156557281, -0.0727326195,  0.3848648469,
            0.8525720202,  0.3378976625, -0.0727326195
        ], dtype=np.float32)

    elif wavelet == "bior1.3":
        # Biorthogonal 1.3 low-pass (decomposition low-pass)
        dec_lo_np = np.array([
            -0.0883883476,  0.0883883476,
            0.7071067812,  0.7071067812,
            0.0883883476, -0.0883883476
        ], dtype=np.float32)

    else:
        raise ValueError(f"Unsupported wavelet: {wavelet}")

    # compute high-pass via standard orthogonal construction:
    # dec_hi[n] = (-1)^n * dec_lo[K-1-n]
    K = dec_lo_np.size
    signs = np.array([(-1) ** n for n in range(K)], dtype=np.float32)
    dec_hi_np = dec_lo_np[::-1] * signs

    # to torch and expand to [C, 1, K] for depthwise conv
    dec_lo = torch.tensor(dec_lo_np, dtype=torch.float32, device=device).view(1, 1, -1).repeat(C, 1, 1)
    dec_hi = torch.tensor(dec_hi_np, dtype=torch.float32, device=device).view(1, 1, -1).repeat(C, 1, 1)

    # Depthwise conv with stride=2 for downsampling
    low_bc = F.conv1d(x_bc, dec_lo, stride=2, groups=C)   # [B, C, L//2]
    high_bc = F.conv1d(x_bc, dec_hi, stride=2, groups=C)  # [B, C, L//2]

    # Back to [B, L//2, C]
    low = low_bc.permute(0, 2, 1)
    high = high_bc.permute(0, 2, 1)

    # Concatenate along time dimension -> [B, L_concat, C]
    out = torch.cat([low, high], dim=1)

    return out


def OLMA(forecasting, label, weight=1.0, weight_c=0.34, weight_t=0.33, weight_w=0.33, wavelet='haar'):
    """
    OLMA, a supervision method that applies frequency domain loss along 
    both the channel and temporal dimensions of time series.
    It is plug-and-play and can be seamlessly integrated into any supervised learning framework.

    Args:
        forecasting: Tensor of shape [B, L, C],
        label: Tensor of shape [B, L, C],
        weight: The overall weight of the OLMA loss,
        weight_c: Weight of channel dimension Fourier Transform loss,
        weight_t: Weight of temporal dimension Fourier Transform loss,
        weight_w: Weight of temporal dimension Wavelet Transform loss,
        wavelet: The choice of the wavelet basis (default Haar).

    Returns:
        OLMA loss.
    """
    # FFT along channel dimension
    pred_fft_c = torch.fft.rfft(forecasting, dim=-1)
    target_fft_c = torch.fft.rfft(label, dim=-1)
    loss_channel = torch.mean(torch.abs(pred_fft_c - target_fft_c))

    # FFT along temporal dimension
    pred_fft_t = torch.fft.rfft(forecasting, dim=1)
    target_fft_t = torch.fft.rfft(label, dim=1)
    loss_temporal = torch.mean(torch.abs(pred_fft_t - target_fft_t))

    # Wavelet low-frequency components (replace FFT in this branch)
    pred_wavelet = Wavelet_Transformer(forecasting, wavelet=wavelet)
    target_wavelet = Wavelet_Transformer(label, wavelet=wavelet)

    # Match sequence length to avoid size mismatch
    min_len = min(pred_wavelet.shape[1], target_wavelet.shape[1])
    pred_wavelet_equal = pred_wavelet[:, :min_len, :]
    target_wavelet_equal = target_wavelet[:, :min_len, :]
    loss_wavelet = torch.mean(torch.abs(pred_wavelet_equal - target_wavelet_equal))

    # Weighted sum of three components
    loss = weight_c * loss_channel + weight_t * loss_temporal + weight_w * loss_wavelet

    return weight * loss