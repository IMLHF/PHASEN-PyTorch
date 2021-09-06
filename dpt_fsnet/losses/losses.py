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

def batchMean_sisnrLoss(est, clean, eps=1e-8):
  batch_sisnr = sisnr(est, clean, eps)
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
