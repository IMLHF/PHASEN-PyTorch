from torch import nn
import torch
import collections
import numpy as np

from ..FLAGS import PARAM
from ..utils import losses
from ..utils import misc_utils
from ..models import conv_stft
from .dpt_blocks.new_model import Net as DPT_FSNET


class Losses(
    collections.namedtuple("Losses",
                           ("sum_loss", "show_losses", "stop_criterion_loss"))):
  pass


class WavFeatures(
    collections.namedtuple("WavFeatures",
                           ("wav_batch", # [N, L]
                            "stft_batch", #[N, 2, F, T]
                            "mag_batch", # [N, F, T]
                            "angle_batch", # [N, F, T]
                            "normed_stft_batch", # [N, F, T]
                            ))):
  pass


class Net(nn.Module):
  def __init__(self, mode, device):
    super(Net, self).__init__()
    self.mode = mode
    self.device = device
    self._net_model = DPT_FSNET(PARAM.frequency_dim, PARAM.dpt_fsnet_width)
    self._stft_fn = conv_stft.ConvSTFT(PARAM.frame_length, PARAM.frame_step, PARAM.fft_length) # [N, 2, F, T]
    self._istft_fn = conv_stft.ConviSTFT(PARAM.frame_length, PARAM.frame_step, PARAM.fft_length) # [N, L]

    if mode == PARAM.MODEL_VALIDATE_KEY or mode == PARAM.MODEL_INFER_KEY:
      # self.eval(True)
      self.to(self.device)
      return

    # other params to save
    self._global_step = 1
    self._start_epoch = 1
    self._nan_grads_batch = 0

    # choose optimizer
    if PARAM.optimizer == "Adam":
      self._optimizer = torch.optim.Adam(self.parameters(), lr=PARAM.learning_rate)
    elif PARAM.optimizer == "RMSProp":
      self._optimizer = torch.optim.RMSprop(self.parameters(), lr=PARAM.learning_rate)

    # for lr warmup
    self._lr_scheduler = None
    if PARAM.use_lr_warmup:
      def warmup(step):
        return misc_utils.warmup_coef(step, warmup_steps=PARAM.warmup_steps)
      self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, warmup)

    # self.train(True)
    self.to(self.device)

  def save_every_epoch(self, ckpt_path):
    self._start_epoch += 1
    torch.save({
                "global_step": self._global_step,
                "start_epoch": self._start_epoch,
                "nan_grads_batch": self._nan_grads_batch,
                "other_state": self.state_dict(),
            }, ckpt_path)

  def load(self, ckpt_path):
    ckpt = torch.load(ckpt_path)
    self._global_step = ckpt["global_step"]
    self._start_epoch = ckpt["start_epoch"]
    self._nan_grads_batch = ckpt["nan_grads_batch"]
    self.load_state_dict(ckpt["other_state"])

  def update_params(self, loss):
    self.zero_grad()
    loss.backward()
    # deal grads

    # grads check nan or inf
    has_nan_inf = 0
    for params in self.parameters():
      if params.requires_grad:
        has_nan_inf += torch.sum(torch.isnan(params.grad))
        has_nan_inf += torch.sum(torch.isinf(params.grad))

    # print('has_nan', has_nan_inf)

    if has_nan_inf == 0:
      self._optimizer.step()
      self._lr_scheduler.step(self._global_step)
      self._global_step += 1
      return
    self._nan_grads_batch += 1

  def forward(self, mixed_wav_batch):
    mixed_wav_batch = mixed_wav_batch.to(self.device)
    mixed_stft_batch = self._stft_fn(mixed_wav_batch) # [N, 2, F, T]
    mixed_stft_real = mixed_stft_batch[:, 0, :, :] # [N, F, T]
    mixed_stft_imag = mixed_stft_batch[:, 1, :, :] # [N, F, T]
    mixed_mag_batch = torch.sqrt(mixed_stft_real**2+mixed_stft_imag**2) # [N, F, T]
    mixed_angle_batch = torch.atan2(mixed_stft_imag, mixed_stft_real) # [N, F, T]
    _N, _F, _T = mixed_mag_batch.size()
    mixed_normed_stft_batch = torch.div(
        mixed_stft_batch, mixed_mag_batch.view(_N, 1, _F, _T)+PARAM.stft_div_norm_eps)
    self.mixed_wav_features = WavFeatures(wav_batch=mixed_wav_batch,
                                          stft_batch=mixed_stft_batch,
                                          mag_batch=mixed_mag_batch,
                                          angle_batch=mixed_angle_batch,
                                          normed_stft_batch=mixed_normed_stft_batch)

    feature_in = self.mixed_wav_features.mixed_stft_batch # [N, 2, F, T]

    est_stft_batch = self._net_model(feature_in) # [N, 2, F, T] -> [N, 2, F, T]

    est_wav_batch = self._istft_fn(est_stft_batch)
    _mixed_wav_length = self.mixed_wav_features.wav_batch.size()[-1]
    est_wav_batch = est_wav_batch[:, :_mixed_wav_length]

    est_stft_real = est_stft_batch[:, 0, :, :] # [N, F, T]
    est_stft_imag = est_stft_batch[:, 1, :, :] # [N, F, T]
    est_mag_batch = torch.sqrt(est_stft_real**2+est_stft_imag**2) # [N, F, T]
    est_angle_batch = torch.atan2(est_stft_imag, est_stft_real) # [N, F, T]
    _N, _F, _T = est_mag_batch.size()
    est_normed_stft_batch = torch.div(
        est_stft_batch, est_mag_batch.view(_N, 1, _F, _T)+PARAM.stft_div_norm_eps)

    return WavFeatures(wav_batch=est_wav_batch,
                       stft_batch=est_stft_batch,
                       mag_batch=est_mag_batch,
                       angle_batch=est_angle_batch,
                       normed_stft_batch=est_normed_stft_batch)

  def get_losses(self, est_wav_features:WavFeatures, clean_wav_batch):
    self.clean_wav_batch = clean_wav_batch.to(self.device)
    self.clean_stft_batch = self._stft_fn(self.clean_wav_batch) # [N, 2, F, T]
    clean_stft_real = self.clean_stft_batch[:, 0, :, :] # [N, F, T]
    clean_stft_imag = self.clean_stft_batch[:, 1, :, :] # [N, F, T]
    self.clean_mag_batch = torch.sqrt(clean_stft_real**2+clean_stft_imag**2) # [N, F, T]
    self.clean_angle_batch = torch.atan2(clean_stft_imag, clean_stft_real) # [N, F, T]
    _N, _F, _T = self.clean_mag_batch.size()
    self.clean_normed_stft_batch = torch.div(
        self.clean_stft_batch, self.clean_mag_batch.view(_N, 1, _F, _T)+PARAM.stft_div_norm_eps)

    est_clean_mag_batch = est_wav_features.mag_batch
    est_clean_stft_batch = est_wav_features.stft_batch
    est_clean_wav_batch = est_wav_features.wav_batch
    est_clean_normed_stft_batch = est_wav_features.normed_stft_batch

    all_losses = list()
    all_losses.extend(PARAM.sum_losses)
    all_losses.extend(PARAM.show_losses)
    all_losses.extend(PARAM.stop_criterion_losses)
    all_losses = set(all_losses)

    self.loss_compressedMag_mse = 0
    self.loss_compressedStft_mse = 0
    self.loss_mag_mse = 0
    self.loss_mag_reMse = 0
    self.loss_stft_mse = 0
    self.loss_stft_reMse = 0
    self.loss_mag_mae = 0
    self.loss_mag_reMae = 0
    self.loss_stft_mae = 0
    self.loss_stft_reMae = 0
    self.loss_wav_L1 = 0
    self.loss_wav_L2 = 0
    self.loss_wav_reL2 = 0
    self.loss_CosSim = 0
    self.loss_SquareCosSim = 0

    # region losses
    if "loss_compressedMag_mse" in all_losses:
      self.loss_compressedMag_mse = losses.batchSum_compressedMag_mse(
          est_clean_mag_batch, self.clean_mag_batch, PARAM.loss_compressedMag_idx)
    if "loss_compressedStft_mse" in all_losses:
      self.loss_compressedStft_mse = losses.batchSum_compressedStft_mse(
          est_clean_mag_batch, est_clean_normed_stft_batch,
          self.clean_mag_batch, self.clean_normed_stft_batch,
          PARAM.loss_compressedMag_idx)


    if "loss_mag_mse" in all_losses:
      self.loss_mag_mse = losses.batchSum_MSE(est_clean_mag_batch, self.clean_mag_batch)
    if "loss_mag_reMse" in all_losses:
      self.loss_mag_reMse = losses.batchSum_relativeMSE(est_clean_mag_batch, self.clean_mag_batch,
                                                        PARAM.relative_loss_epsilon, PARAM.RL_idx)
    if "loss_stft_mse" in all_losses:
      self.loss_stft_mse = losses.batchSum_MSE(est_clean_stft_batch, self.clean_stft_batch)
    if "loss_stft_reMse" in all_losses:
      self.loss_stft_reMse = losses.batchSum_relativeMSE(est_clean_stft_batch, self.clean_stft_batch,
                                                         PARAM.relative_loss_epsilon, PARAM.RL_idx)


    if "loss_mag_mae" in all_losses:
      self.loss_mag_mae = losses.batchSum_MAE(est_clean_mag_batch, self.clean_mag_batch)
    if "loss_mag_reMae" in all_losses:
      self.loss_mag_reMae = losses.batchSum_relativeMAE(est_clean_mag_batch, self.clean_mag_batch,
                                                        PARAM.relative_loss_epsilon)
    if "loss_stft_mae" in all_losses:
      self.loss_stft_mae = losses.batchSum_MAE(est_clean_stft_batch, self.clean_stft_batch)
    if "loss_stft_reMae" in all_losses:
      self.loss_stft_reMae = losses.batchSum_relativeMAE(est_clean_stft_batch, self.clean_stft_batch,
                                                         PARAM.relative_loss_epsilon)


    if "loss_wav_L1" in all_losses:
      self.loss_wav_L1 = losses.batchSum_MAE(est_clean_wav_batch, self.clean_wav_batch)
    if "loss_wav_L2" in all_losses:
      self.loss_wav_L2 = losses.batchSum_MSE(est_clean_wav_batch, self.clean_wav_batch)
    if "loss_wav_reL2" in all_losses:
      self.loss_wav_reL2 = losses.batchSum_relativeMSE(est_clean_wav_batch, self.clean_wav_batch,
                                                       PARAM.relative_loss_epsilon, PARAM.RL_idx)

    if "loss_CosSim" in all_losses:
      self.loss_CosSim = losses.batchMean_CosSim_loss(est_clean_wav_batch, self.clean_wav_batch)
    if "loss_SquareCosSim" in all_losses:
      self.loss_SquareCosSim = losses.batchMean_SquareCosSim_loss(
          est_clean_wav_batch, self.clean_wav_batch)
    # self.loss_stCosSim = losses.batch_short_time_CosSim_loss(est_clean_wav_batch, self.clean_wav_batch,
    #                                                          PARAM.st_frame_length_for_loss,
    #                                                          PARAM.st_frame_step_for_loss)
    # self.loss_stSquareCosSim = losses.batch_short_time_SquareCosSim_loss(est_clean_wav_batch, self.clean_wav_batch,
    #                                                                      PARAM.st_frame_length_for_loss,
    #                                                                      PARAM.st_frame_step_for_loss)
    loss_dict = {
        'loss_compressedMag_mse': self.loss_compressedMag_mse,
        'loss_compressedStft_mse': self.loss_compressedStft_mse,
        'loss_mag_mse': self.loss_mag_mse,
        'loss_mag_reMse': self.loss_mag_reMse,
        'loss_stft_mse': self.loss_stft_mse,
        'loss_stft_reMse': self.loss_stft_reMse,
        'loss_mag_mae': self.loss_mag_mae,
        'loss_mag_reMae': self.loss_mag_reMae,
        'loss_stft_mae': self.loss_stft_mae,
        'loss_stft_reMae': self.loss_stft_reMae,
        'loss_wav_L1': self.loss_wav_L1,
        'loss_wav_L2': self.loss_wav_L2,
        'loss_wav_reL2': self.loss_wav_reL2,
        'loss_CosSim': self.loss_CosSim,
        'loss_SquareCosSim': self.loss_SquareCosSim,
        # 'loss_stCosSim': self.loss_stCosSim,
        # 'loss_stSquareCosSim': self.loss_stSquareCosSim,
    }
    # endregion losses

    # region sum_loss
    sum_loss = 0.0
    sum_loss_names = PARAM.sum_losses
    for i, name in enumerate(sum_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.sum_losses_w) > 0:
        loss_t *= PARAM.sum_losses_w[i]
      sum_loss += loss_t
    # endregion sum_loss

    # region show_losses
    show_losses = []
    show_loss_names = PARAM.show_losses
    for i, name in enumerate(show_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.show_losses_w) > 0:
        loss_t *= PARAM.show_losses_w[i]
      show_losses.append(loss_t)
    show_losses = torch.stack(show_losses)
    # endregion show_losses

    # region stop_criterion_losses
    stop_criterion_losses_sum = 0.0
    stop_criterion_loss_names = PARAM.stop_criterion_losses
    for i, name in enumerate(stop_criterion_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.stop_criterion_losses_w) > 0:
        loss_t *= PARAM.stop_criterion_losses_w[i]
      stop_criterion_losses_sum += loss_t
    # endregion stop_criterion_losses

    return Losses(sum_loss=sum_loss,
                  show_losses=show_losses,
                  stop_criterion_loss=stop_criterion_losses_sum)


  @property
  def global_step(self):
    return self._global_step

  @property
  def start_epoch(self):
    return self._start_epoch

  @property
  def nan_grads_batch(self):
    return self._nan_grads_batch

  @property
  def optimizer_lr(self):
    return self._optimizer.param_groups[0]['lr']


if __name__ == "__main__":
    # calculate MACCs (multiply-accumulation ops) and Parameters
    from thop import profile
    device = 'cpu'
    test_model = DPT_FSNET(257, 64)
    # wav_batch = torch.rand(1, PARAM.sampling_rate).to(device)
    # emb_batch = torch.rand(1, PARAM.speaker_emb_size).to(device)
    wav_feature_in = WavFeatures(wav_batch=torch.rand(1, PARAM.sampling_rate//10).to(device),
                                 stft_batch=torch.rand(1, 2, PARAM.frequency_dim, PARAM.sampling_rate//PARAM.frame_step//10).to(device),  # [N, 2, F, T]
                                 mag_batch=torch.rand(1, PARAM.frequency_dim, PARAM.sampling_rate//PARAM.frame_step//10).to(device),
                                 angle_batch=torch.rand(1, PARAM.frequency_dim, PARAM.sampling_rate//PARAM.frame_step//10).to(device),
                                 normed_stft_batch=torch.rand(1, 2, PARAM.frequency_dim, PARAM.sampling_rate//PARAM.frame_step//10).to(device))
    macs, params = profile(test_model, inputs=(wav_feature_in.stft_batch,))
    print("Config class name: %s\n"%PARAM().config_name())
    # print("model name: %s\n"%PARAM.model_type)
    print("MACCs of processing 1s wav = %.2fM\n"%(macs/1e6))
    print("params = %.2fM\n\n\n"%(params/1e6))
    # del tmp_model