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
