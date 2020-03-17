from torch.utils.data import DataLoader
import numpy as np
import os
import torch

from .data_pipline import data_pipline
from .utils import misc_utils
from .utils import audio
from .utils.assess import core as assess_core
from .models import conv_stft
from .utils import losses
from .FLAGS import PARAM

def test_dataloader():
  noisy_trainset_wav = misc_utils.datasets_dir().joinpath(PARAM.validation_noisy_set)
  clean_trainset_wav = misc_utils.datasets_dir().joinpath(PARAM.validation_clean_set)
  dataset = data_pipline.NoisyCleanDataset(noisy_trainset_wav, clean_trainset_wav)
  dataloader = DataLoader(dataset, batch_size=4,
                          shuffle=True, num_workers=0)
  for i_batch, sample in enumerate(dataloader):
    noisy, clean, noisy_name, clean_name = sample
    print('i_batch', i_batch, noisy_name, clean_name, np.shape(noisy), np.shape(clean))


def wav_through_stft_istft():
  frame_length = 400
  frame_step = 160
  n_fft = 512
  wav_dir = os.path.join("exp", "test", "p232_001.wav")
  wav, sr = audio.read_audio(str(wav_dir))
  print("sr", sr)
  wav_batch = torch.from_numpy(np.array([wav], dtype=np.float32))
  stft_fn = conv_stft.ConvSTFT(frame_length, frame_step, n_fft)
  spec = stft_fn(wav_batch) # [N, 2, F, T]

  istft_fn = conv_stft.ConviSTFT(frame_length, frame_step, n_fft)
  wav2 = istft_fn(spec)

  print(wav_batch.size(), wav2.size())

  wav_np = wav2.numpy()[0][:len(wav)]
  pesq = assess_core.calc_pesq(wav, wav_np, sr)
  sdr = assess_core.calc_sdr(wav, wav_np, sr)
  stoi = assess_core.calc_stoi(wav, wav_np, sr)
  print(pesq, sdr, stoi)

def testCosSim():
  a=torch.randn([12,48000], dtype=torch.float32)
  b=torch.randn([12,48000], dtype=torch.float32)
  a = a / 1e100
  a[:,0] = 1
  # b=torch.randn([12,48000])
  loss = losses.batchMean_CosSim_loss(a,b)
  print(loss.dtype)


if __name__ == "__main__":
  # test_dataloader()
  # wav_through_stft_istft()
  testCosSim()
