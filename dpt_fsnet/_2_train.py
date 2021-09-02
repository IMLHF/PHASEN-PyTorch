import os
import sys
import time
import torch
import collections
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader


from .models import dpt_fsnet
from .data_pipline import data_pipline
from .utils import misc_utils
from .utils import audio
from .sepm import compare
from .FLAGS import PARAM

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


class TrainOutputs(
    collections.namedtuple("TrainOutputs",
                           ("sum_loss", "show_losses",
                            "cost_time", "lr"))):
  pass

grads_nan_time = 0

def train_one_epoch(train_model, train_batch_iter, train_log_file):
  train_model.train()

  s_time = time.time()
  minbatch_time = time.time()
  one_batch_time = time.time()

  avg_sum_loss = None
  avg_show_losses = None
  total_i = PARAM.n_train_set_records//PARAM.batch_size
  for i, batch_in in enumerate(train_batch_iter, 1):
    noisy_batch, clean_batch, _, _ = batch_in
    noisy_batch = noisy_batch.to(train_model.device)
    clean_batch = clean_batch.to(train_model.device)
    est_features = train_model(noisy_batch)
    losses = train_model.get_losses(est_features, clean_batch)
    lr = train_model.optimizer_lr
    train_model.update_params(losses.sum_loss)
    sum_loss = losses.sum_loss.cpu().detach().numpy()
    show_losses = losses.show_losses.cpu().detach().numpy()

    if avg_sum_loss is None:
      avg_sum_loss = sum_loss
      avg_show_losses = show_losses
    else:
      avg_sum_loss += sum_loss
      avg_show_losses += show_losses
    i += 1
    print("\r", end="")
    print(
        "train: %d/%d, cost %.2fs, sum_loss %.4f, show_losses %s, lr %.2e"
        "                  " % (
          i, total_i, time.time()-one_batch_time, sum_loss,
          str(np.round(show_losses, 4)), lr),
        flush=True, end="")
    one_batch_time = time.time()
    if i % PARAM.batches_to_logging == 0:
      print("\r", end="")
      msg = "  Minbatch %04d: sum_loss:%.4f, show_losses:%s, lr:%.2e, time:%ds. \n" % (
              i, avg_sum_loss/i, np.round(avg_show_losses/i, 4), lr, time.time()-minbatch_time,
            )
      minbatch_time = time.time()
      misc_utils.print_log(msg, train_log_file)
    # if i > 10 : break
  print("\r", end="")
  e_time = time.time()
  avg_sum_loss = avg_sum_loss / total_i
  avg_show_losses = avg_show_losses / total_i
  return TrainOutputs(sum_loss=avg_sum_loss,
                      show_losses=np.round(avg_show_losses, 4),
                      cost_time=e_time-s_time,
                      lr=lr)


class EvalOutputs(
    collections.namedtuple("EvalOutputs",
                           ("sum_loss", "show_losses", "cost_time"))):
  pass

def round_lists(lst, rd):
  return [round(n,rd) if type(n) is not list else round_lists(n,rd) for n in lst]

def unfold_list(lst):
  ans_lst = []
  [ans_lst.append(n) if type(n) is not list else ans_lst.extend(unfold_list(n)) for n in lst]
  return ans_lst

def eval_one_epoch(val_model, val_batch_iter):
  val_model.eval()
  # return EvalOutputs(sum_loss=0,
  #                    show_losses=0,
  #                    cost_time=0)

  val_s_time = time.time()
  ont_batch_time = time.time()

  avg_sum_loss = None
  avg_show_losses = None
  total_i = PARAM.n_val_set_records//PARAM.batch_size
  for i, batch_in in enumerate(val_batch_iter, 1):
    with torch.no_grad():
      noisy_batch, clean_batch, _, _ = batch_in
      noisy_batch = noisy_batch.to(val_model.device)
      clean_batch = clean_batch.to(val_model.device)
      # print(noisy_batch.dtype)
      est_features = val_model(noisy_batch)
      losses = val_model.get_losses(est_features, clean_batch)
      sum_loss = losses.sum_loss.cpu().numpy()
      show_losses = losses.show_losses.cpu().numpy()

      # print(np.mean(val_model.clean_mag_batch.cpu().numpy()),
      #       np.std(val_model.clean_mag_batch.cpu().numpy()),
      #       np.mean(val_model.mixed_wav_features.mag_batch.cpu().numpy()),
      #       np.std(val_model.mixed_wav_features.mag_batch.cpu().numpy()), flush=True)

    if avg_sum_loss is None:
      avg_sum_loss = sum_loss
      avg_show_losses = show_losses
    else:
      avg_sum_loss += sum_loss
      avg_show_losses += show_losses
    # if i >5 : break
    print("\r", end="")
    print("validate: %d/%d, cost %.2fs, sum_loss %.4f, show_losses %s"
          "                        " % (
              i, total_i, time.time()-ont_batch_time, sum_loss,
              str(np.round(show_losses, 4))
          ),
          flush=True, end="")
    ont_batch_time = time.time()

  print("\r", end="")
  avg_sum_loss = avg_sum_loss / total_i
  avg_show_losses = avg_show_losses / total_i
  val_e_time = time.time()
  return EvalOutputs(sum_loss=avg_sum_loss,
                     show_losses=np.round(avg_show_losses, 4),
                     cost_time=val_e_time-val_s_time)


class TestOutputs(
    collections.namedtuple("TestOutputs",
                           ("csig", "cbak", "covl", "pesq", "ssnr",
                            "cost_time"))):
  pass

def test_one_epoch(test_model):
  test_model.eval()
  t1 = time.time()
  testset_name = PARAM.test_noisy_sets[0]
  testset_dir = misc_utils.datasets_dir().joinpath(testset_name)
  _dir = misc_utils.enhanced_testsets_save_dir(testset_name)
  if _dir.exists():
    import shutil
    shutil.rmtree(str(_dir))
  _dir.mkdir(parents=True)
  enhanced_save_dir = str(_dir)

  noisy_path_list = list(map(str, testset_dir.glob("*.wav")))
  noisy_num = len(noisy_path_list)
  for i, noisy_path in enumerate(noisy_path_list):
    print("\renhance test wavs: %d/%d" % (i, noisy_num), flush=True, end="")
    noisy_wav, sr = audio.read_audio(noisy_path)
    with torch.no_grad():
      noisy_inputs = torch.from_numpy(np.array([noisy_wav], dtype=np.float32))
      noisy_inputs = noisy_inputs.to(test_model.device)
      est_features = test_model(noisy_inputs)
      noisy_name = Path(noisy_path).stem
      est_wav = est_features.wav_batch.cpu().numpy()[0]
    audio.write_audio(os.path.join(enhanced_save_dir, noisy_name+'_enhanced.wav'),
                      est_wav, PARAM.sampling_rate)
  print("\r                                                               \r", end="", flush=True)

  testset_name, cleanset_name = PARAM.test_noisy_sets[0], PARAM.test_clean_sets[0]
  print("\rCalculate PM %s:" % testset_name, flush=True, end="")
  ref_dir = str(misc_utils.datasets_dir().joinpath(cleanset_name))
  deg_dir = str(misc_utils.enhanced_testsets_save_dir(testset_name))
  res = compare(ref_dir, deg_dir, False)

  pm = np.array([x[1:] for x in res])
  pm = np.mean(pm, axis=0)
  pm = tuple(pm)
  print("\r                                                                    "
        "                                                                      \r", end="")
  t2 = time.time()
  return TestOutputs(csig=pm[0], cbak=pm[1], covl=pm[2], pesq=pm[3], ssnr=pm[4],
                     cost_time=t2-t1)


def main():
  train_log_file = misc_utils.train_log_file_dir()
  ckpt_dir = misc_utils.ckpt_dir()
  hparam_file = misc_utils.hparams_file_dir()
  if not train_log_file.parent.exists():
    os.makedirs(str(train_log_file.parent))
  if not ckpt_dir.exists():
    os.mkdir(str(ckpt_dir))

  misc_utils.save_hparams(str(hparam_file))

  noisy_trainset_wav = misc_utils.datasets_dir().joinpath(PARAM.train_noisy_set)
  clean_trainset_wav = misc_utils.datasets_dir().joinpath(PARAM.train_clean_set)
  noisy_valset_wav = misc_utils.datasets_dir().joinpath(PARAM.validation_noisy_set)
  clean_valset_wav = misc_utils.datasets_dir().joinpath(PARAM.validation_clean_set)
  train_dataset = data_pipline.NoisyCleanDataset(noisy_trainset_wav, clean_trainset_wav)
  val_dataset = data_pipline.NoisyCleanDataset(noisy_valset_wav, clean_valset_wav)
  train_batch_iter = DataLoader(train_dataset, batch_size=PARAM.batch_size,
                                shuffle=True, num_workers=0)
  val_batch_iter = DataLoader(val_dataset, batch_size=PARAM.batch_size*2,
                              shuffle=True, num_workers=0)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # print(device)
  dpt_fsnet_model = dpt_fsnet.Net(PARAM.MODEL_TRAIN_KEY, device)

  ckpt_lst = [str(_dir) for _dir in list(ckpt_dir.glob("*.ckpt"))]
  if len(ckpt_lst) > 0:
    ckpt_lst.sort()
    dpt_fsnet_model.load(ckpt_lst[-1])
    misc_utils.print_log("load ckpt %s\n" % ckpt_lst[-1])
  else:
    for name, param in dpt_fsnet_model.named_parameters():
      print(name, ": ", param.size(), flush=True)

    # region validation before training
    misc_utils.print_log("\n\n", train_log_file)
    misc_utils.print_log("sum_losses: "+str(PARAM.sum_losses)+"\n", train_log_file)
    misc_utils.print_log("show losses: "+str(PARAM.show_losses)+"\n", train_log_file)
    evalOutputs_prev = eval_one_epoch(dpt_fsnet_model, val_batch_iter)
    misc_utils.print_log("                                            "
                         "                                            "
                         "                                         \n",
                         train_log_file, no_time=True)
    val_msg = "PRERUN.val> sum_loss:%.4F, show_losses:%s, Cost itme:%.2Fs.\n" % (
        evalOutputs_prev.sum_loss,
        evalOutputs_prev.show_losses,
        evalOutputs_prev.cost_time)
    misc_utils.print_log(val_msg, train_log_file)

  s_epoch = dpt_fsnet_model.start_epoch
  assert s_epoch > 0, 'start epoch > 0 is required.'
  max_epoch = int(PARAM.max_step / (PARAM.n_train_set_records / PARAM.batch_size))

  for epoch in range(s_epoch, max_epoch+1):
    misc_utils.print_log("\n\n", train_log_file, no_time=True)
    misc_utils.print_log("Epoch %03d/%03d:\n" % (epoch, max_epoch+1), train_log_file)
    misc_utils.print_log("  sum_losses: "+str(PARAM.sum_losses)+"\n", train_log_file)
    misc_utils.print_log("  show_losses: "+str(PARAM.show_losses)+"\n", train_log_file)

    # train
    trainOutputs = train_one_epoch(dpt_fsnet_model, train_batch_iter, train_log_file)
    misc_utils.print_log("  Train     > sum_loss:%.4f, show_losses:%s, lr:%.2e Time:%ds.   \n" % (
        trainOutputs.sum_loss,
        trainOutputs.show_losses,
        trainOutputs.lr,
        trainOutputs.cost_time),
        train_log_file)

    # validation
    evalOutputs = eval_one_epoch(dpt_fsnet_model, val_batch_iter)
    misc_utils.print_log("  Validation> sum_loss%.4f, show_losses:%s, Time:%ds.           \n" % (
        evalOutputs.sum_loss,
        evalOutputs.show_losses,
        evalOutputs.cost_time),
        train_log_file)

    # test
    testOutputs = test_one_epoch(dpt_fsnet_model)
    misc_utils.print_log("  Test      > Csig: %.3f, Cbak: %.3f, Covl: %.3f, pesq: %.3f,"
                         " ssnr: %.4f, Time:%ds.           \n" % (
                             testOutputs.csig, testOutputs.cbak, testOutputs.covl, testOutputs.pesq,
                             testOutputs.ssnr, testOutputs.cost_time),
                         train_log_file)

    # save ckpt
    ckpt_name = PARAM().config_name()+('_iter%04d_trloss%.4f_valloss%.4f_lr%.2e_duration%ds.ckpt' % (
        epoch, trainOutputs.sum_loss, evalOutputs.sum_loss, trainOutputs.lr,
        trainOutputs.cost_time+evalOutputs.cost_time+testOutputs.cost_time))
    dpt_fsnet_model.save_every_epoch(str(ckpt_dir.joinpath(ckpt_name)))
    evalOutputs_prev = evalOutputs
    msg = "  ckpt(%s) saved.\n" % ckpt_name
    misc_utils.print_log(msg, train_log_file)
    msg = "  Nan grad batch %d\n" % dpt_fsnet_model._nan_grads_batch
    misc_utils.print_log(msg, train_log_file)

  # Done
  misc_utils.print_log("\n", train_log_file, no_time=True)
  msg = ("################### Training Done. ###################\n") % dpt_fsnet_model.nan_grads_batch
  misc_utils.print_log(msg, train_log_file)


if __name__ == "__main__":
  misc_utils.initial_run(sys.argv[0].split("/")[-2])
  main()
  """
  run cmd:
  `CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -m xx._2_train`
  """
