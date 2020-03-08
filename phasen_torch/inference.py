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


def enhance_one_wav(model: phasen.PHASEN, wav):
  wav_batch = torch.from_numpy(np.array([wav], dtype=np.float32))
  est_features = model(wav_batch)
  enhanced_wav = est_features.wav_batch.cpu().numpy()[0]
  return enhanced_wav
