'''
Created on Jul 7, 2018

based upon

"A Non-iterative Method for (Re)Construction of Phase from STFT Magnitude"
Zdenek Prusa, Peter Balazs, Peter L. Sondergaard

@author: richard
'''

import numpy as np
from . import rtpghi as rtpghi
from . import pghi as pghi
import scipy.signal as signal
import time

def sine_test():
    f = 10*p.Fs/p.M # fft bin #10
    p.test( 'pure sine test {:4.0f}Hz'.format(f))
    dur = int(2*p.Fs)  #2 seconds
    signal_in = signal.chirp(range(dur), f/p.Fs, dur, f/p.Fs)
    p.signal_to_signal(signal_in)

def pulse_test():
    p.test( 'pulse test')
    magnitude_frames = np.zeros((300,int(p.M/2+1)))
    p.corig_frames = None # kludge to keep from plotting original_phase
    magnitude_frames[20,:]= 1  # pulse at frame 20
    phase_estimated_frames = p.magnitude_to_phase_estimate(magnitude_frames)
    signal_out = p.magphase_frames_to_signal (magnitude_frames, phase_estimated_frames)
    p.plt.plot_waveforms('Signal out', [signal_out])

def sweep_test():
    freq_high = 5000 #Hz
    freq_low = 0
    p.test('sweep test {:.0f}Hz,{:.0f}Hz'.format(freq_low, freq_high))
    dur = int(2*p.Fs)  #swept sine 2 seconds
    method=('linear','quadratic','hyperbolic','logarithmic')[0]
    signal_in = signal.chirp(range(dur), freq_low/p.Fs, dur, freq_high/p.Fs, method=method)
    signal_in2 = signal.chirp(range(dur), freq_high/p.Fs, dur, freq_low/p.Fs, method=method)
    signal_in = np.concatenate([signal_in,signal_in2])
    p.logprint ('duration of sound = {0:10.7} seconds'.format(signal_in.shape[0]/p.Fs))
    p.signal_to_signal(signal_in)

def audio_test():
    for nfile in range(100): # 100 arbitrary file limit
        etime = time.clock()
        song_title, audio_in = p.plt.get_song()
        if audio_in is None:
            break
        stereo = []
        for i in range(audio_in.shape[0]): # channels = 2 for stereo
            p.test(song_title + ' ch{}'.format(i))
            signal_in = audio_in[i]
            signal_out = p.signal_to_signal(signal_in)
            p.plt.plot_waveforms('Signal in, Signal out', [signal_in, signal_out])
            stereo.append( signal_out)
        saved = p.setverbose(True)
        p.test( song_title)
#         saved = p.setverbose(True)
        p.plt.signal_to_file(np.stack(stereo), song_title, override_verbose = True)
        p.logprint('elapsed time = {:8.2f} seconds\n'.format(time.clock()- etime))
        p.setverbose(saved)

def warble_test():
    f1 = 32*p.Fs/p.M # cycles per second
    f2 = 128*p.Fs/p.M
    # set so there is no discontinuity in phase when changing frequencies
    samples_for_2_pi_radians = int(p.Fs/f1)
    p.test('warble test {:.0f}Hz,{:.0f}Hz'.format(f1, f2))
    dur = int(.25*p.Fs/samples_for_2_pi_radians)
    signal_in = []
    for k in range(dur):
        signal_in.append(signal.chirp(range(samples_for_2_pi_radians), f1/p.Fs, samples_for_2_pi_radians, f1/p.Fs))
        signal_in.append(signal.chirp(range(samples_for_2_pi_radians), f2/p.Fs, samples_for_2_pi_radians, f2/p.Fs))
    signal_in = np.concatenate(signal_in)
    p.logprint ('duration of sound = {0:10.7} seconds'.format(signal_in.shape[0]/p.Fs))
    p.signal_to_signal(signal_in)

scale_up = 1
############################  program start ###############################
p = pghi.PGHI(tol = 1e-3, show_frames = 100, time_scale=1/scale_up, freq_scale=scale_up, show_plots = False, verbose=True)


# gl = 2048
# g = signal.windows.hann(gl)
# gamma =gl**2*.25645
# p = pghi.PGHI(tol = 1e-6, show_plots = False, show_frames=10, g=g,gamma = gamma, gl=gl)

# p.setverbose(False)
warble_test()
pulse_test()
sine_test()
sweep_test()
p.setverbose(False)
audio_test()

p = rtpghi.PGHI(tol = 1e-3, show_frames = 100, time_scale=1/scale_up, freq_scale=scale_up, show_plots = False, verbose=True)

# warble_test()
# pulse_test()
# sine_test()
# sweep_test()
p.setverbose(False)
audio_test()







