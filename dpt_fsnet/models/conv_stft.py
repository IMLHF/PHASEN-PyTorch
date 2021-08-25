import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window


def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
  if win_type == 'None' or win_type is None:
    window = np.ones(win_len)
  else:
    window = get_window(win_type, win_len, fftbins=True)**0.5

  N = fft_len
  fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
  real_kernel = np.real(fourier_basis)
  imag_kernel = np.imag(fourier_basis)
  kernel = np.concatenate([real_kernel, imag_kernel], 1).T

  if invers:
    kernel = np.linalg.pinv(kernel).T

  kernel = kernel*window
  kernel = kernel[:, None, :]
  return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None,:,None].astype(np.float32))


class ConvSTFT(nn.Module):

  def __init__(self, win_len, win_inc, fft_len=None, win_type='hanning', pad_end=True,
               feature_type='complex', fix=True):
    super(ConvSTFT, self).__init__()

    if fft_len is None:
      self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
    else:
      self.fft_len = fft_len

    kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
    self.weight = nn.Parameter(kernel, requires_grad=(not fix))
    self.feature_type = feature_type
    self.stride = win_inc
    self.win_len = win_len
    self.dim = self.fft_len
    self.pad_end = pad_end

  def forward(self, inputs):
    '''
    inputs: [batch, n_samples]
    '''
    assert inputs.dim() == 2, 'inputs dims error.'

    if self.pad_end:
      inputs = torch.nn.functional.pad(inputs, [0, self.dim+2])

    inputs = inputs.unsqueeze_(1) # [N, C, L], c=1
    outputs = F.conv1d(
        inputs, self.weight, stride=self.stride,
        # padding=self.dim//2+1,
    )  # [N, 2*F, T]

    if self.feature_type == 'complex':
      shape = outputs.size()
      return outputs.view([shape[0], 2, self.dim//2+1, shape[2]]) # [N, 2, F, T]
    elif self.feature_type == 'polar':
      dim = self.dim//2+1
      real = outputs[:, :dim, :]
      imag = outputs[:, dim:, :]
      mags = torch.sqrt(real**2+imag**2)
      phase = torch.atan2(imag, real)
      return mags, phase

class ConviSTFT(nn.Module):

  def __init__(self, win_len, win_inc, fft_len=None, win_type='hanning', feature_type='complex', fix=True):
    super(ConviSTFT, self).__init__()
    if fft_len is None:
      self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
    else:
      self.fft_len = fft_len
    kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
    self.weight = nn.Parameter(kernel, requires_grad=(not fix))
    self.feature_type = feature_type
    self.win_type = win_type
    self.win_len = win_len
    self.win_inc = win_inc
    self.stride = win_inc
    self.dim = self.fft_len
    self.register_buffer('window', window)
    self.register_buffer('enframe', torch.eye(win_len)[:,None,:])

  def forward(self, inputs, phase=None):
    """
    inputs : [B, 2, F, T] (complex spec) or [B, F, T] (mags)
    phase: [B, F, T] (if not none)
    """

    if phase is not None:
      assert inputs.dim()==3, 'inputs dim error.'
      assert phase.dim()==3, 'pahse dim error'
      real = inputs*torch.cos(phase)
      imag = inputs*torch.sin(phase)
      inputs = torch.cat([real, imag], 1)
    else:
      assert inputs.dim()==4, 'inputs dim error'
      shape = inputs.size() #[N, 2, F, T]
      inputs = inputs.reshape([shape[0], shape[1]*shape[2], shape[3]])
    outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)

    # this is from torch-stft: https://github.com/pseeth/torch-stft
    t = self.window.repeat(1,1,inputs.size(-1))**2
    coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
    #outputs = torch.where(coff == 0, outputs, outputs/coff)
    outputs = outputs/(coff+1e-8) #[N, 1, n_samples]
    shape = outputs.size()
    return outputs.view(shape[0], shape[-1])
