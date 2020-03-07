from torch import nn
import torch
import collections
import numpy as np

from ..FLAGS import PARAM
from ..utils import losses
from ..utils import misc_utils
from ..models import conv_stft


class SelfConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size,
               stride=1, use_bias=True, padding='same', activation=None):
    super(SelfConv2d, self).__init__()
    assert padding.lower() in ['same', 'valid'], 'padding must be same or valid.'
    if padding.lower() == 'same':
      if type(kernel_size) is int:
        padding_nn = kernel_size // 2
      else:
        padding_nn = []
        for kernel_s in kernel_size:
          padding_nn.append(kernel_s // 2)
    self.conv2d_fn = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, bias=use_bias, padding=padding_nn)
    self.act = activation

  def forward(self, feature_in):
    """
    feature_in : [N, C, F, T]
    """
    out = feature_in
    out = self.conv2d_fn(out)
    if self.act is not None:
      out = self.act(out)
    return out


class BatchNormAndActivate(nn.Module):
  def __init__(self, channel, activation=nn.Relu(inplace=True)):
    super(BatchNormAndActivate, self).__init__()
    self.bn_layer = nn.BatchNorm2d(channel)
    self.activate_fn = activation

  def forward(self, fea_in):
    """
    fea_in: [N, C, F, T]
    """
    out = self.bn_layer(fea_in)
    if self.activate_fn is not None:
      out = self.activate_fn(out)
    return out

  def get_bn_weight(self):
    return self.bn_layer.parameters()


class Stream_PreNet(nn.Module):
  def __init__(self, in_channels, out_channels, kernels=[[7, 1], [1, 7]],
               conv2d_activation=None, conv2d_bn=False):
    '''
    channel_out: output channel
    kernels: kernel for layers
    '''
    super(Stream_PreNet, self).__init__()
    self.nn_layers = []
    for i, kernel in enumerate(kernels):
      conv2d = SelfConv2d(in_channels if i==0 else out_channels, out_channels,
                          kernel_size=kernel,
                          activation=(None if conv2d_bn else conv2d_activation),
                          padding="same")
      self.nn_layers.append(conv2d)
      if conv2d_bn:
        bn_fn = BatchNormAndActivate(out_channels, activation=conv2d_activation)
        self.nn_layers.append(bn_fn)


  def forward(self, feature_in):
    '''
    feature_in : [batch, channel_in, F, T]
    return : [batch, channel_out, F, T]
    '''
    if len(self.nn_layers) == 0:
      return feature_in
    out = feature_in
    for layer_fn in self.nn_layers:
      out = layer_fn(out)
    return out


class NodeReshape(nn.Module):
  def __init__(self, shape):
    super(NodeReshape, self).__init__()
    self.shape = shape

  def forward(self, feature_in:torch.Tensor):
    shape = feature_in.size()
    batch = shape[0]
    new_shape = [batch].extend(list(self.shape))
    return feature_in.view(new_shape)


class FrequencyTransformationBlock(nn.Module):
  def __init__(self, frequency_dim, channel_in_out, channel_attention=5):
    super(FrequencyTransformationBlock, self).__init__()
    self.frequency_dim = frequency_dim
    self.channel_out = channel_in_out
    self.att_conv2d_1 = SelfConv2d(channel_in_out, channel_attention, [1, 1],
                                   padding="same")
    self.att_conv2d_1_bna = BatchNormAndActivate(channel_attention)

    # [batch, channel_attention, F, T] -> [batch, channel_attention*F, T]
    self.att_inner_reshape = NodeReshape([frequency_dim * channel_attention, -1])
    self.att_conv1d_2 = nn.Conv1d(frequency_dim*channel_attention, frequency_dim,
                                  9, padding=4)  # [batch, F, T]
    self.att_conv1d_2_bna = BatchNormAndActivate(frequency_dim)
    self.att_out_reshape = NodeReshape([1, frequency_dim, -1])
    self.frequencyFC = nn.Linear(frequency_dim, frequency_dim)

    self.out_conv2d = SelfConv2d(channel_in_out*2, channel_in_out, [1, 1], padding="same")
    self.out_conv2d_bna = BatchNormAndActivate(channel_in_out)

  def forward(self, feature_in, training):
    '''
    feature_n: [batch, channel_in_out, F, T]
    '''
    att_out = self.att_conv2d_1(feature_in)
    att_out = self.att_conv2d_1_bna(att_out)

    att_out = self.att_inner_reshape(att_out)
    att_out = self.att_conv1d_2(att_out)
    att_out = self.att_conv1d_2_bna(att_out)
    att_out = self.att_out_reshape(att_out)

    atted_out = torch.mul(feature_in, att_out) # [batch, channel_in_out, F, T]
    atted_out_T = torch.transpose(atted_out, 2, 3) # [batch, channel_in_out, T, F]
    ffc_out_T = self.frequencyFC(atted_out_T)
    ffc_out = torch.transpose(ffc_out_T, 2, 3) # [batch, channel_in_out, F, T]
    concated_out = torch.cat([feature_in, ffc_out], 1) # [batch, channel_in_out*2, F, T]

    out = self.out_conv2d(concated_out)
    out = self.out_conv2d_bna(out)
    return out


class InfoCommunicate(nn.Module):
  def __init__(self, channel_in, channel_out, activate_fn=nn.Tanh()):
    super(InfoCommunicate, self).__init__()
    self.conv2d = SelfConv2d(
        channel_in, channel_out, [1, 1], padding="same")
    self.activate_fn = activate_fn

  def forward(self, feature_x1, feature_x2):
    # feature_x1: [batch, channel_out, F, T]
    # feature_x2: [batch, channel_in, F, T]
    # return: [batch, channel_out, F, T]
    out = self.conv2d(feature_x2)
    if self.activate_fn is not None:
      out = self.activate_fn(out)

    out_multiply = torch.mul(feature_x1, out)
    return out_multiply


class TwoStreamBlock(nn.Module):
  def __init__(self, frequency_dim, channel_in_out_A, channel_in_out_P):
    super(TwoStreamBlock, self).__init__()
    self.sA1_pre_FTB = FrequencyTransformationBlock(
        frequency_dim, channel_in_out_A)
    self.sA2_conv2d = SelfConv2d(
        channel_in_out_A, channel_in_out_A, [5, 5], padding="same")
    self.sA2_conv2d_bna = BatchNormAndActivate(channel_in_out_A)
    self.sA3_conv2d = SelfConv2d(
        channel_in_out_A, channel_in_out_A, [1, 25], padding="same")
    self.sA3_conv2d_bna = BatchNormAndActivate(channel_in_out_A)
    self.sA4_conv2d = SelfConv2d(
        channel_in_out_A, channel_in_out_A, [5, 5], padding="same")
    self.sA4_conv2d_bna = BatchNormAndActivate(channel_in_out_A)
    self.sA5_post_FTB = FrequencyTransformationBlock(
        frequency_dim, channel_in_out_A)
    self.sA6_info_communicate = InfoCommunicate(
        channel_in_out_P, channel_in_out_A)

    # [batch, C, F, T]
    self.sP1_conv2d_before_LN = nn.LayerNorm([channel_in_out_P, frequency_dim, 1])
    self.sP1_conv2d = SelfConv2d(
        channel_in_out_P, channel_in_out_P, [3, 5], padding="same")
    self.sP2_conv2d_before_LN = nn.LayerNorm([channel_in_out_P, frequency_dim, 1])
    self.sP2_conv2d = SelfConv2d(
        channel_in_out_P, channel_in_out_P, [1, 25], padding="same")
    self.sP3_info_communicate = InfoCommunicate(
        channel_in_out_A, channel_in_out_P)

  def forward(self, feature_sA, feature_sP):
    # Stream A
    sA_out = feature_sA
    sA_out = self.sA1_pre_FTB(sA_out)
    sA_out = self.sA2_conv2d(sA_out)
    sA_out = self.sA2_conv2d_bna(sA_out)
    sA_out = self.sA3_conv2d(sA_out)
    sA_out = self.sA3_conv2d_bna(sA_out)
    sA_out = self.sA4_conv2d(sA_out)
    sA_out = self.sA4_conv2d_bna(sA_out)
    sA_out = self.sA5_post_FTB(sA_out)

    # Strean P
    sP_out = feature_sP
    sP_out = self.sP1_conv2d_before_LN(sP_out)
    sP_out = self.sP1_conv2d(sP_out)
    sP_out = self.sP2_conv2d_before_LN(sP_out)
    sP_out = self.sP2_conv2d(sP_out)

    # information communication
    sA_fin_out = self.sA6_info_communicate(sA_out, sP_out)
    sP_fin_out = self.sP3_info_communicate(sP_out, sA_out)

    return sA_fin_out, sP_fin_out


class StreamAmplitude_PostNet(nn.Module):
  def __init__(self, frequency_dim, channel_sA):
    super(StreamAmplitude_PostNet, self).__init__()
    self.p1_conv2d = SelfConv2d(channel_sA, 8, [1, 1],
                                activation=nn.Sigmoid(),
                                padding="same") #[N, 8, F, T]

    uni_rnn_units = 600
    self.p1_reshape = NodeReshape([frequency_dim * 8, -1])
    self.p2_blstm = nn.LSTM(frequency_dim * 8, uni_rnn_units, batch_first=True, bidirectional=True)

    self.p3_dense = nn.Sequential(nn.Linear(uni_rnn_units * 2, 600), nn.ReLU(inplace=True))
    self.p4_dense = nn.Sequential(nn.Linear(600, 600), nn.ReLU(inplace=True))
    self.out_dense = nn.Sequential(nn.Linear(600, 600), nn.Sigmoid())

  def call(self, feature_sA, training):
    '''
    return [batch, T, F]
    '''
    out = feature_sA
    out = self.p1_conv2d(out)
    out = self.p1_reshape(out) # [N, 8*F, T]
    out = torch.transpose(out, 1, 2) # [N, T, 8*F]
    out = self.p2_blstm(out)[0] # [N, T, 2*600]
    out = self.p3_dense(out)
    out = self.p4_dense(out)
    out = self.out_dense(out)
    return out


class StreamPhase_PostNet(nn.Module):
  def __init__(self, channel_sP):
    super(StreamPhase_PostNet, self).__init__()
    self._layers.append(SelfConv2d(
        channel_sP, 2, [1, 1], padding="same"))

  def forward(self, feature_sP:torch.Tensor):
    '''
    return [batch, T, F]->complex
    '''
    out = feature_sP
    for layer_fn in self._layers:
      out = layer_fn(out)
    # out: [batch, 2, F, T]
    out_real = out[:, 0, :, :]
    out_imag = out[:, 1, :, :]
    out_angle = out_imag.atan2_(out_real)
    out[:, 0, :, :] = torch.cos(out_angle)
    out[:, 1, :, :] = torch.sin_(out_angle)
    return out


class WavFeatures(
    collections.namedtuple("WavFeatures",
                           ("wav_batch", # [N, L]
                            "stft_batch", #[N, 2, F, T]
                            "mag_batch", # [N, F, T]
                            "angle_batch", # [N, F, T]
                            ))):
  pass


class NET_PHASEN_OUT(
    collections.namedtuple("NET_PHASEN_OUT",
                           ("mag_mask", "normalized_complex_phase"))):
  pass


class NetPHASEN(nn.Module):
  def __init__(self):
    super(NetPHASEN, self).__init__()
    self.streamA_prenet = Stream_PreNet(
        2, PARAM.channel_A, kernels=PARAM.prenet_A_kernels,
        conv2d_bn=True, conv2d_activation=nn.ReLU(inplace=True))
    self.streamP_prenet = Stream_PreNet(
        2, PARAM.channel_P, PARAM.prenet_P_kernels)
    self.layers_TSB = []
    for i in range(1, PARAM.n_TSB+1):
      tsb_t = TwoStreamBlock(
          PARAM.frequency_dim, PARAM.channel_A, PARAM.channel_P)
      self.layers_TSB.append(tsb_t)
    self.streamA_postnet = StreamAmplitude_PostNet(
        PARAM.frequency_dim, PARAM.channel_A)
    self.streamP_postnet = StreamPhase_PostNet(PARAM.channel_P)

  def forward(self, mixed_wav_features:WavFeatures):
    '''
    Args:
      mixed_wav_features
    Return :
      mag_batch[batch, time, fre]->real,
      normalized_complex_phase[batch, time, fre, 2]->(real, imag)
    '''
    inputs = mixed_wav_features.stft_batch # [N, 2, F, T]

    sA_out = self.streamA_prenet(inputs)  # [batch, Ca, f, t]
    sP_out = self.streamP_prenet(inputs)  # [batch, Cp, f, t]
    for tsb in self.layers_TSB:
      sA_out, sP_out = tsb(sA_out, sP_out)
    sA_out = self.streamA_postnet(sA_out)  # [batch, f, t]
    sP_out = self.streamP_postnet(sP_out)  # [batch, 2, f, t]

    est_mask = sA_out  # [batch, f, t]
    normed_complex_phase = sP_out  # [batch, 2, f, t], (real, imag)
    return NET_PHASEN_OUT(mag_mask=est_mask,
                          normalized_complex_phase=normed_complex_phase)


class Losses(
    collections.namedtuple("Losses",
                           ("sum_loss", "show_losses", "stop_criterion_loss"))):
  pass


class PHASEN(nn.Module):
  def __init__(self,
               mode):
    super(PHASEN, self).__init__()
    self.mode = mode
    self._net_model = NetPHASEN()
    self._stft_fn = conv_stft.ConvSTFT(PARAM.frame_length, PARAM.frame_step, PARAM.fft_length) # [N, 2, F, T]
    self._istft_fn = conv_stft.ConviSTFT(PARAM.frame_length, PARAM.frame_step, PARAM.fft_length) # [N, L]

    if mode == PARAM.MODEL_VALIDATE_KEY or mode == PARAM.MODEL_INFER_KEY:
      self.eval(True)
      return

    # global_step
    self._global_step = 1

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

    self.train(True)

  def save(self, ckpt_path):
    torch.save({
                "global_step": self._global_step,
                "other_state": self.state_dict(),
            }, ckpt_path)

  def load(self, ckpt_path):
    ckpt = torch.load(ckpt_path)
    self._global_step = ckpt['global_step']
    self.load_state_dict(ckpt['other_state'])

  def forward(self, mixed_wav_batch):
    mixed_wav_batch = mixed_wav_batch
    mixed_stft_batch = self._stft_fn(mixed_wav_batch) # [N, 2, F, T]
    mixed_stft_real = mixed_stft_batch[:, 0, :, :] # [N, F, T]
    mixed_stft_imag = mixed_stft_batch[:, 1, :, :] # [N, F, T]
    mixed_mag_batch = torch.sqrt(mixed_stft_real**2+mixed_stft_imag**2) # [N, F, T]
    mixed_angle_batch = torch.atan2(mixed_stft_imag, mixed_stft_real) # [N, F, T]
    self.mixed_wav_features = WavFeatures(wav_batch=mixed_wav_batch,
                                          stft_batch=mixed_stft_batch,
                                          mag_batch=mixed_mag_batch,
                                          angle_batch=mixed_angle_batch)

    feature_in = self.mixed_wav_features # [N, 2, F, T]

    net_phasen_out = self.net_model(feature_in)

    est_clean_mag_batch = torch.mul(
        self.mixed_wav_features.mag_batch, net_phasen_out.mag_mask)  # [batch, F, T]
    mag_shape = est_clean_mag_batch.size()
    est_complexPhase_batch = net_phasen_out.normalized_complex_phase # [bathch, 2, F, T]
    est_clean_stft_batch = torch.mul(
        est_clean_mag_batch.view([mag_shape[0], 1, mag_shape[1], mag_shape[2]]),
        est_complexPhase_batch)
    est_clean_wav_batch = self._istft_fn(est_clean_stft_batch)
    _mixed_wav_length = self.mixed_wav_features.wav_batch.size()[-1]
    est_clean_wav_batch = est_clean_wav_batch[:, :_mixed_wav_length]

    return WavFeatures(wav_batch=est_clean_wav_batch,
                       stft_batch=est_clean_stft_batch,
                       mag_batch=est_clean_mag_batch,
                       angle_batch=None)

  def get_losses(self, est_wav_features:WavFeatures, clean_wav_batch):
    if clean_wav_batch is not None:
      self.clean_wav_batch = clean_wav_batch
      self.clean_stft_batch = self._stft_fn(self.clean_wav_batch) # [N, 2, F, T]
      clean_stft_real = self.clean_stft_batch[:, 0, :, :] # [N, F, T]
      clean_stft_imag = self.clean_stft_batch[:, 1, :, :] # [N, F, T]
      self.clean_mag_batch = torch.sqrt(clean_stft_real**2+clean_stft_imag**2) # [N, F, T]
      # self.clean_angle_batch = torch.atan2(clean_stft_imag, clean_stft_real) # [N, F, T]

    est_clean_mag_batch = est_wav_features.mag_batch
    est_clean_stft_batch = est_wav_features.stft_batch
    est_clean_wav_batch = est_wav_features.wav_batch


    # region losses
    self.loss_compressedMag_mse = losses.batchSum_compressedMag_mse(est_clean_mag_batch,
                                                                    self.clean_mag_batch,
                                                                    PARAM.loss_compressedMag_idx)
    self.loss_compressedStft_mse = losses.batchSum_compressedStft_mse(est_clean_stft_batch,
                                                                      self.clean_stft_batch,
                                                                      PARAM.loss_compressedMag_idx)


    self.loss_mag_mse = losses.batchSum_MSE(est_clean_mag_batch, self.clean_mag_batch)
    self.loss_mag_reMse = losses.batchSum_relativeMSE(est_clean_mag_batch, self.clean_mag_batch,
                                                      PARAM.relative_loss_epsilon, PARAM.RL_idx)
    self.loss_stft_mse = losses.batchSum_MSE(est_clean_stft_batch, self.clean_stft_batch)
    self.loss_stft_reMse = losses.batchSum_relativeMSE(est_clean_stft_batch, self.clean_stft_batch,
                                                       PARAM.relative_loss_epsilon, PARAM.RL_idx)


    self.loss_mag_mae = losses.batchSum_MAE(est_clean_mag_batch, self.clean_mag_batch)
    self.loss_mag_reMae = losses.batchSum_relativeMAE(est_clean_mag_batch, self.clean_mag_batch,
                                                      PARAM.relative_loss_epsilon)
    self.loss_stft_mae = losses.batchSum_MAE(est_clean_stft_batch, self.clean_stft_batch)
    self.loss_stft_reMae = losses.batchSum_relativeMAE(est_clean_stft_batch, self.clean_stft_batch,
                                                       PARAM.relative_loss_epsilon)


    self.loss_wav_L1 = losses.batchSum_MAE(est_clean_wav_batch, self.clean_wav_batch)
    self.loss_wav_L2 = losses.batchSum_MSE(est_clean_wav_batch, self.clean_wav_batch)
    self.loss_wav_reL2 = losses.batchSum_relativeMSE(est_clean_wav_batch, self.clean_wav_batch,
                                                     PARAM.relative_loss_epsilon, PARAM.RL_idx)

    self.loss_CosSim = losses.batch_CosSim_loss(
        est_clean_wav_batch, self.clean_wav_batch)
    self.loss_SquareCosSim = losses.batch_SquareCosSim_loss(
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
  def optimizer_lr(self):
    return self._optimizer.param_groups[0]['lr']
