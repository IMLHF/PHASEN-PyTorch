import tensorflow as tf
import torch
import sys
import tensorflow.contrib.slim as slim
import time
from distutils import version
from pathlib import Path
import os

from ..FLAGS import PARAM

def tf_wav2mag(batch_wav, frame_length, frame_step, n_fft):
  cstft = tf.signal.stft(batch_wav, frame_length, frame_step, fft_length=n_fft, pad_end=True)
  feature = tf.math.abs(cstft)
  return feature


def tf_wav2stft(batch_wav, frame_length, frame_step, n_fft):
  cstft = tf.signal.stft(batch_wav, frame_length, frame_step, fft_length=n_fft, pad_end=True)
  return cstft


def tf_stft2wav(batch_stft, frame_length, frame_step, n_fft):
  signals = tf.signal.inverse_stft(batch_stft, frame_length, frame_step, fft_length=n_fft,
                                   window_fn=tf.signal.inverse_stft_window_fn(frame_step))
  return signals


def initial_run(config_name):
  assert config_name == PARAM().config_name(), (
      "config name error: dir.%s|FLAG.%s." % (config_name, PARAM().config_name()))
  check_torch_version()
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
  print_hparams()


def test_code_out_dir():
  _dir = Path(PARAM.root_dir).joinpath("exp", "test")
  return _dir


def enhanced_testsets_save_dir(testset_name):
  exp_config_name_dir = exp_configName_dir()
  return exp_config_name_dir.joinpath('enhanced_testsets', testset_name)


def hparams_file_dir():
  exp_config_name_dir = exp_configName_dir()
  return exp_config_name_dir.joinpath('hparam')


def ckpt_dir():
  exp_config_name_dir = exp_configName_dir()
  return exp_config_name_dir.joinpath('ckpt')


def test_log_file_dir(testset_name):
  str_snr = "%s.test.log" % testset_name
  log_dir_ = log_dir()
  return log_dir_.joinpath(str_snr)


def train_log_file_dir():
  log_dir_ = log_dir()
  return log_dir_.joinpath('train.log')


def log_dir():
  exp_config_name_dir = exp_configName_dir()
  return exp_config_name_dir.joinpath('log')


def exp_configName_dir():
  return Path(PARAM.root_dir).joinpath('exp', PARAM().config_name())


def datasets_dir():
  return Path(PARAM.root_dir).joinpath(PARAM.datasets_name)


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
  '''Noam scheme learning rate decay
  init_lr: initial learning rate. scalar.
  global_step: scalar.
  warmup_steps: scalar. During warmup_steps, learning rate increases
      until it reaches init_lr.
  '''
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def show_variables(vars_):
  slim.model_analyzer.analyze_vars(vars_, print_info=True)
  sys.stdout.flush()


def show_all_variables():
  model_vars = tf.trainable_variables()
  show_variables(model_vars)


def print_log(msg, log_file=None, no_time=False, no_prt=False):
  if log_file is not None:
    log_file = str(log_file)
  if not no_time:
      time_stmp = "%s | " % time.ctime()
      msg = time_stmp+msg
  if not no_prt:
    print(msg, end='', flush=True)
  if log_file:
    with open(log_file, 'a+') as f:
        f.write(msg)


def check_tensorflow_version():
  # LINT.IfChange
  min_tf_version = PARAM.min_TF_version
  # LINT.ThenChange(<pwd>/nmt/copy.bara.sky)
  if not (version.LooseVersion(tf.__version__) == version.LooseVersion(min_tf_version)):
    raise EnvironmentError("Tensorflow version must be '%s'" % min_tf_version)


def check_torch_version():
  # LINT.IfChange
  min_torch_version = PARAM.min_Torch_version
  # LINT.ThenChange(<pwd>/nmt/copy.bara.sky)
  if not (version.LooseVersion(torch.__version__) == version.LooseVersion(min_torch_version)):
    raise EnvironmentError("PyTorch version must be '%s'" % min_torch_version)


def save_hparams(f):
  f = open(f, 'a+')
  from .. import FLAGS
  self_dict = FLAGS.PARAM.__dict__
  self_dict_keys = self_dict.keys()
  f.writelines('FLAGS.PARAM:\n')
  supper_dict = FLAGS.BaseConfig.__dict__
  for key in sorted(supper_dict.keys()):
    if key in self_dict_keys:
      f.write('%s:%s\n' % (key,self_dict[key]))
    else:
      f.write('%s:%s\n' % (key,supper_dict[key]))
  f.write('--------------------------\n\n')

  f.write('Short hparams:\n')
  [f.write("%s:%s\n" % (key, self_dict[key])) for key in sorted(self_dict_keys)]
  f.write('--------------------------\n\n')


def print_hparams(short=True):
  from .. import FLAGS
  self_dict = FLAGS.PARAM.__dict__
  self_dict_keys = self_dict.keys()
  print('\n--------------------------\n')
  print('Short hparams:')
  [print("%s:%s" % (key, self_dict[key])) for key in sorted(self_dict_keys)]
  print('--------------------------\n')
  if not short:
    print('FLAGS.PARAM:')
    supper_dict = FLAGS.BaseConfig.__dict__
    for key in sorted(supper_dict.keys()):
      if key in self_dict_keys:
        print('%s:%s' % (key,self_dict[key]))
      else:
        print('%s:%s' % (key,supper_dict[key]))
    print('--------------------------\n')
    print('Short hparams:')
    [print("%s:%s" % (key, self_dict[key])) for key in sorted(self_dict_keys)]
    print('--------------------------\n')
