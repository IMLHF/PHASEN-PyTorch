class StaticKey(object):
  MODEL_TRAIN_KEY = 'train'
  MODEL_VALIDATE_KEY = 'validation'
  MODEL_INFER_KEY = 'infer'

  def config_name(self): # config_name
    return self.__class__.__name__

class BaseConfig(StaticKey):
  VISIBLE_GPU = "0"
  root_dir = '/home/lhf/worklhf/TorchPHASEN/'
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

  channel_A = 96
  channel_P = 48
  prenet_A_kernels = [[7,1], [1, 7]]
  prenet_P_kernels = [[3,5], [1,25]]
  n_TSB = 3
  frequency_dim = 257
  loss_compressedMag_idx = 0.3

  stream_A_feature_type = "stft" # "stft" | "mag"
  stream_P_feature_type = "stft" # "stft" | "normed_stft"
  stft_norm_method = "atan2" # atan2 | div
  stft_div_norm_eps = 1e-5 # for stft_norm_method=div

  clip_grads = False
  net_out_type = 'mask'
  sA_out_act = "sigmoid"


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
  channel_A = 96
  channel_P = 48
  n_TSB = 3


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
  channel_A = 96
  channel_P = 48
  n_TSB = 3
  stream_A_feature_type = "stft"
  stream_P_feature_type = "normed_stft"
  stft_norm_method = "div"
  stft_div_norm_eps = 1e-7
  # test_noisy_sets = ['noisy_trainset_wav']
  # test_clean_sets = ['clean_trainset_wav']

class se_phasen_0092(p40): # running v100
  '''
  phasen 0092
  loss_compressedMag_mse + loss_compressedStft_mse
  '''
  sum_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  sum_losses_w = []
  show_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse",
                 "loss_CosSim", "loss_mag_mse", "loss_stft_mse"]
  show_losses_w = []
  stop_criterion_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  stop_criterion_losses_w = []
  channel_A = 96
  channel_P = 48
  n_TSB = 3
  stream_A_feature_type = "stft"
  stream_P_feature_type = "normed_stft"
  stft_norm_method = "div"
  stft_div_norm_eps = 1e-7

  train_noisy_set = 'noisy_trainset_wav_phasen009_en'
  train_clean_set = 'clean_trainset_wav'
  validation_noisy_set = 'noisy_testset_wav_phasen009_en'
  validation_clean_set = 'clean_testset_wav'
  test_noisy_sets = ['noisy_testset_wav_phasen009_en']
  test_clean_sets = ['clean_testset_wav']

class se_phasen_0093(p40): # running v100
  '''
  phasen 0092
  loss_compressedMag_mse + loss_compressedStft_mse
  '''
  sum_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  sum_losses_w = []
  show_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse",
                 "loss_CosSim", "loss_mag_mse", "loss_stft_mse"]
  show_losses_w = []
  stop_criterion_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  stop_criterion_losses_w = []
  channel_A = 96
  channel_P = 48
  n_TSB = 3
  stream_A_feature_type = "stft"
  stream_P_feature_type = "normed_stft"
  stft_norm_method = "div"
  stft_div_norm_eps = 1e-7

  train_noisy_set = 'noisy_trainset_wav_phasen009_en'
  train_clean_set = 'clean_trainset_wav'
  validation_noisy_set = 'noisy_testset_wav_phasen009_en'
  validation_clean_set = 'clean_testset_wav'
  test_noisy_sets = ['noisy_testset_wav_phasen009_en']
  test_clean_sets = ['clean_testset_wav']
  net_out_type = 'magres'
  sA_out_act = 'None'

class se_phasen_010(p40): # done v100
  '''
  phasen 010
  loss_compressedMag_mse + loss_compressedStft_mse
  '''
  sum_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  sum_losses_w = []
  show_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse",
                 "loss_CosSim", "loss_mag_mse", "loss_stft_mse"]
  show_losses_w = []
  stop_criterion_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  stop_criterion_losses_w = []
  channel_A = 96
  channel_P = 48
  n_TSB = 3
  stream_A_feature_type = "stft"
  stream_P_feature_type = "stft"
  stft_norm_method = "div"
  stft_div_norm_eps = 1e-7


class se_phasen_011(p40): # done v100
  '''
  phasen 011
  loss_compressedMag_mse + loss_compressedStft_mse
  '''
  sum_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  sum_losses_w = []
  show_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse",
                 "loss_CosSim", "loss_mag_mse", "loss_stft_mse"]
  show_losses_w = []
  stop_criterion_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  stop_criterion_losses_w = []
  channel_A = 96
  channel_P = 48
  n_TSB = 3
  stream_A_feature_type = "mag"
  stream_P_feature_type = "normed_stft"
  stft_norm_method = "div"
  stft_div_norm_eps = 1e-7

class se_phasen_012(p40): # running v100
  '''
  phasen 012
  loss_compressedMag_mse + loss_compressedStft_mse
  replace StreamA of TSB with BLSTM*1 ->worse
  remove 5*5 conv in streamA of TSB ->worse
  '''
  sum_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  sum_losses_w = []
  show_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse",
                 "loss_CosSim", "loss_mag_mse", "loss_stft_mse"]
  show_losses_w = []
  stop_criterion_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  stop_criterion_losses_w = []
  channel_A = 96
  channel_P = 48
  n_TSB = 3
  stream_A_feature_type = "stft"
  stream_P_feature_type = "normed_stft"
  stft_norm_method = "div"
  stft_div_norm_eps = 1e-7

class se_phasen_013(p40): # running v100
  '''
  phasen 013
  loss_compressedMag_mse + loss_compressedStft_mse
  '''
  sum_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  sum_losses_w = []
  show_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse",
                 "loss_CosSim", "loss_mag_mse", "loss_stft_mse"]
  show_losses_w = []
  stop_criterion_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  stop_criterion_losses_w = []
  channel_A = 96
  channel_P = 48
  n_TSB = 3
  stream_A_feature_type = "stft"
  stream_P_feature_type = "normed_stft"
  stft_norm_method = "div"
  stft_div_norm_eps = 1e-7
  batch_size = 6


PARAM = se_phasen_009 ###

# CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 python -m se_phasen_0093._2_train
