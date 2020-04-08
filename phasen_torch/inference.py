import torch
import collections
import numpy as np

from .models import phasen
from .utils import misc_utils
from .utils.phase_reconstruction import rtpghi
from .FLAGS import PARAM

phase_reconstructor = None


def build_model(ckpt_dir=None):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  phasen_model = phasen.PHASEN(PARAM.MODEL_INFER_KEY, device)
  phasen_model.eval()
  if ckpt_dir is not None:
    phasen_model.load(ckpt_dir)
  else:
    ckpt_lst = [str(_dir) for _dir in list(misc_utils.ckpt_dir().glob("*.ckpt"))]
    ckpt_lst.sort()
    phasen_model.load(ckpt_lst[-1])
  return phasen_model


def enhance_one_wav(model: phasen.PHASEN, wav, phase_type=0):
  '''
  phase_tyoe: [0: est, 1: nisy, 2: form mag]
  '''
  wav_batch = torch.from_numpy(np.array([wav], dtype=np.float32))
  len_wav = len(wav)
  with torch.no_grad():
    est_features = model(wav_batch)
    if phase_type == 0:
      enhanced_wav = est_features.wav_batch.cpu().numpy()[0]
    elif phase_type == 1:
      enhanced_mag = est_features.mag_batch.unsqueeze(1) # [B, 1, F, T]
      noisy_phase = model.mixed_wav_features.normed_stft_batch # [B, 2, F, T]
      enhanced_stft = enhanced_mag * noisy_phase
      enhanced_wav = model._istft_fn(enhanced_stft).cpu().numpy()[0][:len_wav]
      # print('noisy_phase', flush=True)
    elif phase_type == 2:
      enhanced_wav = est_features.wav_batch.cpu().numpy()[0]
      wav_len = len(enhanced_wav)
      global phase_reconstructor
      if phase_reconstructor is None:
        phase_reconstructor = rtpghi.PGHI(redundancy=8, M=PARAM.fft_length,
                                          # gl=PARAM.frame_length,
                                          verbose=False,
                                          Fs=PARAM.sampling_rate)
      rec_wav = phase_reconstructor.signal_to_signal(enhanced_wav)
      enhanced_wav = rec_wav[:wav_len]
      assert wav_len==len(enhanced_wav), 'wav length error.'

  return enhanced_wav
