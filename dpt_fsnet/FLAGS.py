class StaticKey(object):
  MODEL_TRAIN_KEY = 'train'
  MODEL_VALIDATE_KEY = 'validation'
  MODEL_INFER_KEY = 'infer'

  def config_name(self): # config_name
    return self.__class__.__name__

class BaseConfig(StaticKey):
  VISIBLE_GPU = "0"
  root_dir = '/root/worklhf/SE_VoiceBankDEMAND/'
  # datasets_name = 'vctk_musan_datasets'
  datasets_name = 'noisy_datasets_16k'
  '''
  # dir to store log, model and results files:
  $root_dir/$datasets_name: datasets dir
  $root_dir/exp/$config_name/log: logs(include tensorboard log)
  $root_dir/exp/$config_name/ckpt: ckpt
  $root_dir/exp/$config_name/enhanced_testsets: enhanced results
  $root_dir/exp/$config_name/hparams
  '''

  # min_TF_version = "1.14.0"
  min_Torch_version = "1.0.0"


  train_noisy_set = 'noisy_trainset_wav'
  train_clean_set = 'clean_trainset_wav'
  validation_noisy_set = 'noisy_testset_wav'
  validation_clean_set = 'clean_testset_wav'
  test_noisy_sets = ['noisy_testset_wav']
  test_clean_sets = ['clean_testset_wav']

  n_train_set_records = 11572
  n_val_set_records = 824
  n_test_set_records = 824

  train_val_wav_seconds = 3.0

  batch_size = 12

  relative_loss_epsilon = 0.1
  RL_idx = 2.0
  st_frame_length_for_loss = 512
  st_frame_step_for_loss = 256
  sampling_rate = 16000
  frame_length = 400
  frame_step = 160
  fft_length = 512
  optimizer = "Adam" # "Adam" | "RMSProp"
  learning_rate = 0.0005
  max_gradient_norm = 5.0

  GPU_RAM_ALLOW_GROWTH = True
  GPU_PARTION = 0.97

  max_step = 40000
  batches_to_logging = 200000

  use_lr_warmup = True # true: lr warmup; false: lr halving
  warmup_steps = 6000. # for (use_lr_warmup == true)

  # melMat: tf.contrib.signal.linear_to_mel_weight_matrix(129,129,8000,125,3900)
  # plt.pcolormesh
  # import matplotlib.pyplot as plt

  """
  @param losses:
  see phasen.py : PHASEN.get_losses()
  """
  sum_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  sum_losses_w = []
  show_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  show_losses_w = []
  stop_criterion_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  stop_criterion_losses_w = []

  dpt_fsnet_width = 64
  frequency_dim = 257
  loss_compressedMag_idx = 0.3

  stft_div_norm_eps = 1e-7 # for stft_norm_method=div



class p40(BaseConfig):
  # GPU_PARTION = 0.27
  root_dir = '/home/zhangwenbo5/lihongfeng/TorchPHASEN'


class test001(p40): # done v100
  '''
  phasen 001
  loss_compressedMag_mse + loss_compressedStft_mse
  '''
  sum_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  sum_losses_w = []
  show_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse", "loss_CosSim", "loss_stft_mse"]
  show_losses_w = []
  stop_criterion_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  stop_criterion_losses_w = []


class se_phasen_009(p40): # done v100
  '''
  phasen 009
  loss_compressedMag_mse + loss_compressedStft_mse
  '''
  sum_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  sum_losses_w = []
  show_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse",
                 "loss_CosSim", "loss_mag_mse", "loss_stft_mse"]
  show_losses_w = []
  stop_criterion_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  stop_criterion_losses_w = []
  stft_div_norm_eps = 1e-7
  # test_noisy_sets = ['noisy_trainset_wav']
  # test_clean_sets = ['clean_trainset_wav']


PARAM = se_phasen_009 ###

# CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 python -m se_phasen_0093._2_train
