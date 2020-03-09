import torch
import collections
import numpy as np

from .models import phasen
from .utils import misc_utils
from .FLAGS import PARAM


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


def enhance_one_wav(model: phasen.PHASEN, wav, use_noisy_phase=False):
  wav_batch = torch.from_numpy(np.array([wav], dtype=np.float32))
  len_wav = len(wav)
  with torch.no_grad():
    est_features = model(wav_batch)
    if not use_noisy_phase:
      enhanced_wav = est_features.wav_batch.cpu().numpy()[0]
    else:
      enhanced_mag = est_features.mag_batch.unsqueeze(1) # [B, 1, F, T]
      noisy_phase = model.mixed_wav_features.normed_stft_batch # [B, 2, F, T]
      enhanced_stft = enhanced_mag * noisy_phase
      enhanced_wav = model._istft_fn(enhanced_stft).cpu().numpy()[0][:len_wav]
      # print('noisy_phase', flush=True)
  return enhanced_wav
