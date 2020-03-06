from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
from pathlib import Path
import torch

from ..utils import audio
from ..FLAGS import PARAM


class NoisyCleanDataset(Dataset):

    def __init__(self, noisy_path, clean_path):
      """
      noisy_path: noisy wavs path
      clean_path: clean wavs path
      """
      noisy_path = Path(noisy_path)
      clean_path = Path(clean_path)
      self.noisy_list = list(map(str, noisy_path.glob("*.wav")))
      self.clean_list = list(map(str, clean_path.glob("*.wav")))
      self.noisy_list.sort()
      self.clean_list.sort()
      self.dataset_len = len(self.noisy_list)

    def __len__(self):
      return self.dataset_len

    def __getitem__(self, idx):
      noisy_dir = self.noisy_list[idx]
      clean_dir = self.clean_list[idx]
      noisy, nsr = audio.read_audio(noisy_dir)
      clean, csr = audio.read_audio(clean_dir)
      assert nsr == csr, "sample rate error."
      wav_len = int(PARAM.train_val_wav_seconds*PARAM.sampling_rate)
      # noisy = audio.repeat_to_len(noisy, wav_len)
      # clean = audio.repeat_to_len(clean, wav_len)
      noisy, clean = audio.repeat_to_len_2(noisy, clean, wav_len, True)
      return torch.from_numy(noisy), torch.from_numpy(clean), Path(noisy_dir).name, Path(clean_dir).name
