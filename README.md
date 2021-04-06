# PHASEN

implentation of paper [Yin, D., Luo, C., Xiong, Z., & Zeng, W. (2020, April). Phasen: A phase-and-harmonics-aware speech enhancement network. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 05, pp. 9458-9465).](https://ojs.aaai.org/index.php/AAAI/article/download/6489/6345).

## Prepare for running

1) Running `bash ./phasen_torch/_1_perprocess.sh` to prepare data.

2) Change "root_dir" parameter in phasen_torch/FLAGS.py to the root of the project. For example "root_dir = /home/user/PHASEN-PYTORCH".

3) Ensure "PARAM = PHASEN_009" is set in last line of phasen_torch/FLAGS.py.

4) Running `cp phasen_torch PHASEN_009 -r` to create the Experiment config code dir.

## Train

Running `python -m PHASEN_009._2_train` to start training of exp config "PHASEN_009".

## Evaluate

Running `python -m PHASEN_009._3_enhance_testsets` to get the metrics of Experiment "PHASEN_009". The last ckpt is selected as the default ckpt to load. Alse, you can use `--ckpt` to specify the path of ckpt.

## More

See "phasen_torch/_1_preprocess.sh", "phasen_torch/_2_train.py" and "phasen_torch/_3_enhance_testsets.py".

## results

The code has basically reproduced the performance in the PHASEN paper (Exp.ID: PHASEN_009). The experimental results are as follows.

|Name|Csig|Cbak|Covl|PESQ|SegSNR|LSD|ESTOI(%)|other||SNR|
|-----------|----|----|----|----|----|----|----|------|-|----|
noisy|3.357 |2.453 |2.649 |1.994 |1.710 |8.253 |78.67%||||8.7104|
PHASEN (torch)||||||||||||
PHASEN_001 (ckpt36)|4.046 |3.477 |3.439 |2.816 |10.397 |||stft+stft|||19.5243|
PHASEN_001 (ckpt36 noisy_phase)|4.031 |3.432 |3.419 |2.796 |9.888 ||||||18.9004|
PHASEN_002 (ckpt36)|4.052 |3.469 |3.464 |2.877 |10.113 |||remse+cos|||19.2635|
PHASEN_002 (ckpt36 noisy_phase)|4.162 |3.401 |3.489 |2.790 |9.106 ||||||18.3013|
PHASEN_003 (ckpt25)|4.108 |3.515 |3.502 |2.885 |10.529 |||mag+stft|||19.6491|
PHASEN_004 (ckpt26)|4.143 |3.524 |3.542 |2.929 |10.353 |||mag+normStft|||19.4845|
PHASEN_005 (ckpt32)|4.160 |3.528 |3.558 |2.934 |10.269 |||stft+normStft|||19.3847|
PHASEN_fix005 (ckpt39)|4.192 |3.528 |3.574 |2.935 |10.198 |||||||
PHASEN_007 (ckpt37)|4.185 |3.539 |3.588 |2.961 |10.141 |||div normstft 1e-6||||
PHASEN_008 (ckpt34)|4.185 |3.540 |3.572 |2.935 |10.373 |||div normstft 1e-5||||
PHASEN_009 (ckpt41)|4.212 |3.557 |3.613 |2.988 |10.287 |||div normstft 1e-7 stft+normStft||||
PHASEN_009 (ckpt41 noisy_phase)|4.181 |3.500 |3.570 |2.935 |9.777 |||||||
PHASEN_009 (ckpt41 rtpghi_phase)|3.624 |2.378 |2.922 |2.278 |-1.660 |||||||
PHASEN_009 (ckpt41 pghi_phase)|4.135 |2.798 |3.539 |2.934 |-1.184 |||||||
PHASEN_010 (ckpt36)|4.142 |3.527 |3.541 |2.921 |10.310 |||div normstft 1e-7 stft+stft||||
PHASEN_011 (ckpt34)|4.186 |3.549 |3.590 |2.966 |10.293 |||div normstft 1e-7 mag+normstft||||
