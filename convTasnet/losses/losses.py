import torch
import torch.nn.functional as F

def vec_dot_mul(y1, y2):
  dot_mul = torch.sum(torch.mul(y1, y2), dim=-1)
  # print('dot', dot_mul.size())
  return dot_mul

def vec_normal(y):
  normal_ = torch.sqrt(torch.sum(y**2, dim=-1))
  # print('norm',normal_.size())
  return normal_

def mag_fn(real, imag):
  return torch.sqrt(real**2+imag**2)

def batchSum_MSE(y1, y2, _idx=2):
  loss = (y1-y2)**_idx
  loss = torch.mean(torch.sum(loss, 0))
  return loss

def batchSum_compressedMag_mse(y1, y2, compress_idx, eps=1e-5):
  """
  y1>=0: real, [batch, F, T]
  y2>=0: real, [batch, F, T]
  """
  y1 = torch.pow(y1+eps, compress_idx)
  y2 = torch.pow(y2+eps, compress_idx)
  loss = batchSum_MSE(y1, y2)
  return loss

def batchSum_compressedStft_mse(est_mag, est_normstft, clean_mag, clean_normstft, compress_idx, eps=1e-5):
  """
  est_mag:                real, [batch, F, T]
  est_normstft:   (real, imag), [batch, 2, F, T]
  clean_mag:              real, [batch, F, T]
  clean_normstft: (real, imag), [batch, 2, F, T]
  """
  # compress_idx = 1.0
  est_abs_cpr = torch.pow(est_mag+eps, compress_idx).unsqueeze_(1) # [batch, 1, F, T]
  clean_abs_cpr = torch.pow(clean_mag+eps, compress_idx).unsqueeze_(1)

  est_cpr_stft = est_abs_cpr * est_normstft
  clean_cpr_stft = clean_abs_cpr * clean_normstft
  loss = batchSum_MSE(est_cpr_stft, clean_cpr_stft)
  return loss

def batchSum_stftmLoss(est_stft, clean_stft):
  '''
  est_stft:   (real, imag), [batch, 2, F, T]
  clean_stft: (real, imag), [batch, 2, F, T]
  '''
  abst = torch.abs(clean_stft) - torch.abs(est_stft)
  tmp = torch.abs(abst[:,0,:,:] + abst[:,1,:,:]) #[B, 1, F, T]
  loss = torch.mean(torch.sum(tmp, 0))
  return loss

def batchSum_relativeMSE(y1, y2, RL_epsilon, index_=2.0):
  # y1, y2 : [batch, F, T]
  relative_loss = torch.abs(y1-y2)/(torch.abs(y1)+torch.abs(y2)+RL_epsilon)
  loss = torch.pow(relative_loss, index_)
  loss = torch.mean(torch.sum(loss, 0))
  return loss

def batchSum_MAE(y1, y2):
  loss = torch.mean(torch.sum(torch.abs(y1-y2), 0))
  return loss

def batchSum_relativeMAE(y1, y2, RL_epsilon):
  # y1, y2 : [batch, F, T]
  relative_loss = torch.abs(y1-y2)/(torch.abs(y1)+torch.abs(y2)+RL_epsilon)
  loss = torch.mean(torch.sum(relative_loss, 0))
  return loss

def batchMean_CosSim_loss(est, ref): # -cos
  '''
  est, ref: [batch, ..., n_sample]
  '''
  # print(est.size(), ref.size(), flush=True)
  cos_sim = - torch.div(vec_dot_mul(est, ref), # [batch, ...]
                        torch.mul(vec_normal(est), vec_normal(ref)))
  loss = torch.mean(cos_sim)
  return loss

def batchMean_SquareCosSim_loss(est, ref): # -cos^2
  # print('23333')
  loss_s1 = - torch.div(vec_dot_mul(est, ref)**2,  # [batch, ...]
                        torch.mul(vec_dot_mul(est, est), vec_dot_mul(ref, ref)))
  loss = torch.mean(loss_s1)
  return loss

def sisnr(x, s, eps):
    """
    Arguments:
    x: separated signal, N x S tensor
    s: reference signal, N x S tensor
    Return:
    sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(x_zm * s_zm, dim=-1,
                  keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True) ** 2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

def calc_sisnr_torch(estimation, origin, mask=None):
    """
    batch-wise SDR caculation for one audio file on pytorch Variables.
    estimation: (batch, nsample)
    origin: (batch, nsample)
    mask: optional, (batch, nsample), binary
    """

    if mask is not None:
        origin = origin * mask
        estimation = estimation * mask

    origin_power = torch.pow(origin, 2).sum(1, keepdim=True) + 1e-8  # (batch, 1)

    scale = torch.sum(origin*estimation, 1, keepdim=True) / origin_power  # (batch, 1)

    est_true = scale * origin  # (batch, nsample)
    est_res = estimation - est_true  # (batch, nsample)

    true_power = torch.pow(est_true, 2).sum(1)
    res_power = torch.pow(est_res, 2).sum(1)

    return 10*torch.log10(true_power) - 10*torch.log10(res_power)  # (batch, 1)

def batchMean_sisnrLoss(est, clean, eps=1e-8):
  batch_sisnr = sisnr(est, clean, eps)
  # print(batch_sisnr.shape)
  return -torch.mean(batch_sisnr)

def batchMean_sisnrLossV2(est, clean):
  batch_sisnr = calc_sisnr_torch(est, clean)
  # print(batch_sisnr.shape)
  return -torch.mean(batch_sisnr)

# def batchMean_short_time_CosSim_loss(est, ref, st_frame_length, st_frame_step): # -cos
#   st_est = tf.signal.frame(est, frame_length=st_frame_length, # [batch, frame, st_wav]
#                            frame_step=st_frame_step, pad_end=True)
#   st_ref = tf.signal.frame(ref, frame_length=st_frame_length,
#                            frame_step=st_frame_step, pad_end=True)
#   loss = batchMean_CosSim_loss(st_est, st_ref)
#   return loss

# def batchMean_short_time_SquareCosSim_loss(est, ref, st_frame_length, st_frame_step): # -cos^2
#   st_est = tf.signal.frame(est, frame_length=st_frame_length, # [batch, frame, st_wav]
#                            frame_step=st_frame_step, pad_end=True)
#   st_ref = tf.signal.frame(ref, frame_length=st_frame_length,
#                            frame_step=st_frame_step, pad_end=True)
#   loss = batchMean_SquareCosSim_loss(st_est, st_ref)
#   return loss


def cal_si_snr(source, estimate_source):
    """Calculate SI-SNR.

    Arguments:
    ---------
    source: [T, B, C],
        Where B is batch size, T is the length of the sources, C is the number of sources
        the ordering is made so that this loss is compatible with the class PitWrapper.

    estimate_source: [T, B, C]
        The estimated source.

    Example:
    ---------
    >>> import numpy as np
    >>> x = torch.Tensor([[1, 0], [123, 45], [34, 5], [2312, 421]])
    >>> xhat = x[:, (1, 0)]
    >>> x = x.unsqueeze(-1).repeat(1, 1, 2)
    >>> xhat = xhat.unsqueeze(1).repeat(1, 2, 1)
    >>> si_snr = -cal_si_snr(x, xhat)
    >>> print(si_snr)
    tensor([[[ 25.2142, 144.1789],
             [130.9283,  25.2142]]])
    """
    EPS = 1e-8
    assert source.size() == estimate_source.size()
    device = estimate_source.device.type

    source_lengths = torch.tensor(
        [estimate_source.shape[0]] * estimate_source.shape[1], device=device
    )
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    num_samples = (
        source_lengths.contiguous().reshape(1, -1, 1).float()
    )  # [1, B, 1]
    mean_target = torch.sum(source, dim=0, keepdim=True) / num_samples
    mean_estimate = (
        torch.sum(estimate_source, dim=0, keepdim=True) / num_samples
    )
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = zero_mean_target  # [T, B, C]
    s_estimate = zero_mean_estimate  # [T, B, C]
    # s_target = <s', s>s / ||s||^2
    dot = torch.sum(s_estimate * s_target, dim=0, keepdim=True)  # [1, B, C]
    s_target_energy = (
        torch.sum(s_target ** 2, dim=0, keepdim=True) + EPS
    )  # [1, B, C]
    proj = dot * s_target / s_target_energy  # [T, B, C]
    # e_noise = s' - s_target
    e_noise = s_estimate - proj  # [T, B, C]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    si_snr_beforelog = torch.sum(proj ** 2, dim=0) / (
        torch.sum(e_noise ** 2, dim=0) + EPS
    )
    si_snr = 10 * torch.log10(si_snr_beforelog + EPS)  # [B, C]

    return si_snr.unsqueeze(0)

def get_mask(source, source_lengths):
    """
    Arguments
    ---------
    source : [T, B, C]
    source_lengths : [B]

    Returns
    -------
    mask : [T, B, 1]

    Example:
    ---------
    >>> source = torch.randn(4, 3, 2)
    >>> source_lengths = torch.Tensor([2, 1, 4]).int()
    >>> mask = get_mask(source, source_lengths)
    >>> print(mask)
    tensor([[[1.],
             [1.],
             [1.]],
    <BLANKLINE>
            [[1.],
             [0.],
             [1.]],
    <BLANKLINE>
            [[0.],
             [0.],
             [1.]],
    <BLANKLINE>
            [[0.],
             [0.],
             [1.]]])
    """
    T, B, _ = source.size()
    mask = source.new_ones((T, B, 1))
    for i in range(B):
        mask[source_lengths[i] :, i, :] = 0
    return mask


if __name__ == "__main__":
  ref = torch.randn(1, 16000)
  est = torch.randn(1, 16000)
  a = sisnr(est, ref, 1e-8)
  b = calc_sisnr_torch(est, ref)
  c = cal_si_snr(ref.unsqueeze(0).permute(2, 1, 0), est.unsqueeze(0).permute(2, 1, 0))
  print(a, b, c)
