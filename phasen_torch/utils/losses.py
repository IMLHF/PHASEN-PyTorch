import torch

def vec_dot_mul(y1, y2):
  dot_mul = torch.sum(torch.mul(y1, y2), dim=-1)
  return dot_mul

def vec_normal(y):
  normal_ = torch.sqrt(torch.sum(y**2, dim=-1))
  return normal_

def mag_fn(real, imag):
  return torch.sqrt(real**2+imag**2)

def batchSum_MSE(y1, y2, _idx=2.0):
  loss = torch.pow(y1-y2, _idx)
  loss = torch.mean(torch.sum(loss, 0))
  return loss

def batchSum_compressedMag_mse(y1, y2, compress_idx):
  """
  y1>=0: real, [batch, F, T]
  y2>=0: real, [batch, F, T]
  """
  y1 = torch.pow(y1, compress_idx)
  y2 = torch.pow(y2, compress_idx)
  loss = batchSum_MSE(y1, y2)
  return loss

def batchSum_compressedStft_mse(y1, y2, compress_idx):
  """
  y1: (real, imag), [batch, 2, F, T]
  y2: (real, imag), [batch, 2, F, T]
  """
  y1_real = y1[:,:1,:,:] # [batch, 1, F, T]
  y1_imag = y1[:,1:,:,:] # [batch, 1, F, T]
  y2_real = y2[:,:1,:,:] # [batch, 1, F, T]
  y2_imag = y2[:,1:,:,:] # [batch, 1, F, T]
  y1_abs_cpr = torch.pow(mag_fn(y1_real, y1_imag), compress_idx) # [batch, 1, F, T]
  y2_abs_cpr = torch.pow(mag_fn(y2_real, y2_imag), compress_idx)
  y1_angle = torch.atan2(y1_imag, y1_real) # [batch, 1, F, T]
  y2_angle = torch.atan2(y2_imag, y2_real)
  y1_cpr_real = y1_abs_cpr * torch.cos(y1_angle)
  y1_cpr_imag = y1_abs_cpr * torch.sin(y1_angle)
  y2_cpr_real = y2_abs_cpr * torch.cos(y2_angle)
  y2_cpr_imag = y2_abs_cpr * torch.sin(y2_angle)
  loss = batchSum_MSE(y1_cpr_real, y2_cpr_real) + batchSum_MSE(y1_cpr_imag, y2_cpr_imag)
  loss = loss * 0.5
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
  cos_sim = - torch.div(vec_dot_mul(est, ref), # [batch, ...]
                        torch.mul(vec_normal(est), vec_normal(ref)))
  loss = torch.mean(cos_sim)
  return loss

def batchMean_SquareCosSim_loss(est, ref): # -cos^2
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
