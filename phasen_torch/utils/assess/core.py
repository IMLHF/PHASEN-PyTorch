# -*- coding: utf-8 -*-
"""
@author: PengChuan
这一部分是语音信号的评价指标，用来评估语音信号降噪的质量，判断结果好坏
    pesq：perceptual evaluation of speech quality，语音质量听觉评估
    stoi：short time objective intelligibility，短时客观可懂度，尤其在低SNR下，可懂度尤其重要
    ssnr: segmental SNR，分段信噪比(时域指标)，它是参考信号和信号差的比值，衡量的是降噪程度
"""

import os
import tempfile
import numpy as np
import librosa
import platform
from pystoi.stoi import stoi
from mir_eval.separation import bss_eval_sources

from .. import audio

# import pesq binary
PESQ_PATH = os.path.split(os.path.realpath(__file__))[0]
if 'Linux' in platform.system():
    PESQ_PATH = os.path.join(PESQ_PATH, 'pesq.ubuntu16.exe')
else:
    PESQ_PATH = os.path.join(PESQ_PATH, 'pesq.win10.exe')


def calc_pesq(ref_sig, deg_sig, samplerate, is_file=False):

    if 'Windows' in platform.system():
        raise NotImplementedError

    if is_file:
        output = os.popen('%s +%d %s %s' % (PESQ_PATH, samplerate, ref_sig, deg_sig))
        msg = output.read()
    else:
        tmp_ref = tempfile.NamedTemporaryFile(suffix='.wav', delete=True)
        tmp_deg = tempfile.NamedTemporaryFile(suffix='.wav', delete=True)
        # librosa.output.write_wav(tmp_ref.name, ref_sig, samplerate)
        # librosa.output.write_wav(tmp_deg.name, deg_sig, samplerate)
        audio.write_audio(tmp_ref.name, ref_sig, samplerate)
        audio.write_audio(tmp_deg.name, deg_sig, samplerate)
        output = os.popen('%s +%d %s %s' % (PESQ_PATH, samplerate, tmp_ref.name, tmp_deg.name))
        msg = output.read()
        tmp_ref.close()
        tmp_deg.close()
        # os.unlink(tmp_ref.name)
        # os.unlink(tmp_deg.name)
    score = msg.split('Prediction : PESQ_MOS = ')
    # print(msg)
    # exit(0)
    # print(score)
    if len(score)<=1:
      print('calculate error.')
      return 2.0
    return float(score[1][:-1])


def calc_stoi(ref_sig, deg_sig, samplerate):
  return stoi(ref_sig, deg_sig, samplerate)


def calc_sdr(ref_sig, deg_sig, samplerate):
    """Calculate Source-to-Distortion Ratio(SDR).
    NOTE: one wav or batch wav.
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T] or [T], src_ref and src_deg must be same dimention.
        src_deg: numpy.ndarray, [C, T] or [T], reordered by best PIT permutation
    Returns:
        SDR
    """
    sdr, sir, sar, popt = bss_eval_sources(ref_sig, deg_sig)
    return sdr[0]
