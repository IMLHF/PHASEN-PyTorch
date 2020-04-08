'''
Created on Jul 7, 2018

based upon

"A Non-iterative Method for (Re)Construction of Phase from STFT Magnitude"
Zdenek Prusa, Peter Balazs, Peter L. Sondergaard

@author: richard lyman
'''
import numpy as np
import heapq
import scipy.signal as signal
from . import pghi_plot
from scipy import ndimage


dtype = np.float64

class PGHI(object):
    '''
    implements the Phase Gradient Heap Integration - PGHI algorithm
    '''

    def __init__(self, redundancy=8, time_scale=1, freq_scale=1, M=2048, gl=None, g=None,
                 tol=1e-6, lambdasqr=None, gamma=None, h=.01, plt=None, pre_title='',
                 show_plots=False,  show_frames=25, verbose=True, Fs=44100):
        '''
        Parameters
            redundancy
                number of hops per window
            time_scale
                multiplier to lengthen/shorten time, higher number is slower output
            freq_scale
                multiplier to expand/contract frequency scale
            M
                number of samples in for each FFT calculation
                measure: samples
            gl length of the sampling window
                measure: samples
            g
                windowing function of shape (gl,)
            lambdasqr
                constant for windowing function
                measure: samples**2
            gamma
                alternative to lambdasqr
                measure 2*pi*samples**2
            tol
                small signal relative magnitude filtering size
                measure: filtering height/maximum magnitude height
            h
                the relative height of the Gaussian window function at edges
                of the window, h = 1 mid window
            pre_title
                string to prepend to the file names when storing plots and sound files
            show_plots
                if True, each plot window becomes active and must be closed to continue
                the program. Handy for rotating the plot with the cursor for 3d plots
                if False, plots are saved to the ./pghi_plots sub directory
            show_frames
                The number of frames to plot on each side of the algorithm start point
            verbose
                boolean, if True then save output to ./pghi_plots directory

            Fs
                sampling frequency
                measure - samples per second
        Example
            p = pghi.PGHI(redundancy=8, M=2048,tol = 1e-6, show_plots = False, show_frames=20)
        '''
        if gl is None: gl = M
        if gamma is not None:
            lambdasqr = gamma/(2*np.pi)
        if g is None:
            # Auger, Motin, Flandrin #19
            lambda_ = (-gl**2/(8*np.log(h)))**.5
            lambdasqr = lambda_**2
            gamma = 2*np.pi*lambdasqr
            g=np.array(signal.windows.gaussian(2*gl+1, lambda_*2, sym=False), dtype = dtype)[1:2*gl+1:2]

        self.redundancy,self.time_scale,self.freq_scale,self.M,self.tol,self.lambdasqr,self.g,self.gl,h, self.pre_title,self.verbose,self.Fs, self.gamma = redundancy,time_scale,freq_scale, M,tol,lambdasqr,g,gl,h,pre_title,verbose,Fs, gamma

        self.M2 = int(self.M/2) + 1

        self.a_s = int(self.M/redundancy)
        self.a_a = int(self.a_s/time_scale)
        self.magnitude =np.zeros((3,self.M2))
        self.phase =np.zeros((3,self.M2))
        self.fgrad =np.zeros((3,self.M2))
        self.tgrad =np.zeros((3,self.M2))
        self.logs =np.zeros((3,self.M2))
        self.original_phase = np.zeros((3,self.M2))
        self.corig = None
        self.plt = pghi_plot.Pghi_Plot(show_plots = show_plots,  show_frames = show_frames, pre_title=pre_title, logfile='log_rtpghi.txt')

        self.setverbose(verbose)
        if lambdasqr is None: self.logprint('parameter error: must supply lambdasqr and g')
        self.logprint('a_a(analysis time hop size) = {} samples'.format(self.a_a))
        self.logprint('a_s(synthesis time hop size) = {} samples'.format(self.a_s))
        self.logprint('M, samples per frame = {}'.format(M))
        self.logprint('tol, small signal filter tolerance ratio = {}'.format(tol))
        self.logprint('lambdasqr = {:9.4f} 2*pi*samples**2  '.format(self.lambdasqr))
        self.logprint('gamma = {:9.4f} 2*pi*samples**2  '.format(self.gamma))
        self.logprint('h, window height at edges = {} relative to max height'.format(h))
        self.logprint('fft bins = {}'.format(self.M2))
        self.logprint('redundancy = {}'.format(redundancy))
        self.logprint('time_scale = {}'.format(time_scale))
        self.logprint('freq_scale = {}'.format(freq_scale))
        self.plt.plot_waveforms("Window Analysis", [self.g])

        denom = 0    # calculate the synthesis window
        self.gsynth = np.zeros_like(self.g, dtype=dtype)
        for l in range(int(self.gl)):
            denom = 0
            for n in range(-redundancy, redundancy+1):
                dl = l-n*self.a_s
                if dl >=0 and dl < self.M:
                    denom += self.g[dl]**2
            self.gsynth[l] = self.g[l]/denom
        self.plt.plot_waveforms("Window Synthesis", [self.gsynth])

    def setverbose(self, verbose):
        saved_d = self.plt.verbose
        self.plt.verbose = verbose
        return saved_d

    def test(self, title):
        self.plt.pre_title = title
        self.logprint ('\n'+title)

    def logprint(self, txt):
        self.plt.logprint(txt)

    def clear(self):
        self.corig= None

    def dxdw(self,x):
        ''' return the derivative of x with respect to frequency'''
        xp = np.pad(x,1,mode='edge')
#         dw = (np.multiply(3,(xp[1:-1,:-2]) + np.multiply(2,xp[1:-1,1:-1]) + np.multiply(3,xp[1:-1,2:])) - np.multiply(6,(xp[1:-1,:-2] + xp[1:-1,1:-1] + xp[1:-1,2:])))/6
        dw = (xp[2:]-xp[:-2])/2
        return dw

    def dxdt(self,x):
        ''' return the derivative of x with respect to time'''
        xp = np.pad(x,1,mode='edge')
#         dt = (np.multiply(3,(xp[:-2,1:-1]) + np.multiply(2,xp[1:-1,1:-1]) + np.multiply(3,xp[2:,1:-1])) - np.multiply(6,(xp[:-2,1:-1] + xp[1:-1,1:-1] + xp[2:,1:-1])))/6
        dt = (xp[1,1:-1]-xp[1,1:-1])/(2)

        return dt

    def magnitude_to_phase_estimate(self, magnitude):
        '''
            run the hop by hop magnitude to phase algorithm through the
            entire sound sample to produce graphs
        '''
        original_phase = np.zeros_like(magnitude)
        if self.plt.verbose:         # for debugging
            self.debug_count=0
            try:
                original_phase = np.angle(self.corig_frames)
            except:
                pass
            self.q_errors=[]
        phase, fgrad, tgrad = [],[],[]

        for n in range(magnitude.shape[0]):
#             self.mask = np.roll(self.mask,-1,axis=0)
#             self.mask[2] = magnitude[n] > (self.tol*np.max(magnitude[n]))
#             print('STEP')
            p, f, t = self.magnitude_to_phase_estimatex(magnitude[n], original_phase[n])
            phase.append(p)
            fgrad.append(f)
            tgrad.append(t)

        mask = magnitude > (self.tol*np.max(magnitude) )
        phase = np.stack(phase)
        tgrad = np.stack(tgrad)
        fgrad = np.stack(fgrad)

        if self.plt.verbose:
            nprocessed = np.sum(np.where(mask,1,0))
            self.logprint ('magnitudes processed above threshold tolerance={}, magnitudes rejected below threshold tolerance={}'.format(nprocessed, magnitude.size-nprocessed) )
            self.plt.plot_3d('magnitude', [magnitude], mask=mask)
            self.plt.plot_3d('fgrad',[fgrad], mask=mask)
            self.plt.plot_3d('tgrad',[tgrad], mask=mask)
            self.plt.plot_3d('Phase estimated', [phase], mask=mask)
            if original_phase is not None:
                self.plt.plot_3d('Phase original', [original_phase], mask=mask)
                self.plt.plot_3d('Phase original, Phase estimated', [(original_phase) %(2*np.pi), ( phase) %(2*np.pi)], mask=mask)
                self.plt.colorgram('Phase original minus Phase estimated', np.abs((original_phase) %(2*np.pi) -( phase) %(2*np.pi)), mask=mask)
                self.plt.quiver('phase errors', self.q_errors)
        return phase

    def magnitude_to_phase_estimatex(self, magnitude, original_phase):
        ''' estimate the phase frames from the magnitude
        parameter:
            magnitude
                numpy array containing the real absolute values of the
                magnitudes of each FFT frame.
                shape (n,m) where n is the frame step and
                m is the frequency step
        return
            estimated phase of each fft coefficient
                shape (n,m) where n is the frame step and
                m is the frequency step
                measure: radians per sample
        '''

        N = magnitude.shape[0]
        M2, M, a_a = self.M2, self.M, self.a_a
        wbin = 2*np.pi/self.M
        self.magnitude = np.roll(self.magnitude,-1,axis=0)
        self.phase = np.roll(self.phase,-1,axis=0)
        self.fgrad = np.roll(self.fgrad,-1,axis=0)
        self.tgrad = np.roll(self.tgrad,-1,axis=0)
        self.logs = np.roll(self.logs,-1,axis=0)
        self.original_phase = np.roll(self.original_phase,-1,axis=0)
        self.magnitude[2] = magnitude
        self.original_phase[2] = original_phase
        eps=np.finfo(np.float64).eps
        self.logs[2] = np.log(magnitude + eps)

        # alternative
#         fmul = self.lambdasqr*wbin/a

        fmul = self.gamma/(a_a * M)
        self.tgradplus = (2*np.pi*a_a/M)*np.arange(M2)
        self.tgrad[2] = self.dxdw(self.logs[2])/fmul + self.tgradplus

        self.fgradplus =    np.pi
        self.fgrad[1] = - fmul*self.dxdt(self.logs) + self.fgradplus

        h=[]
        # print(np.shape(magnitude))
        mask = magnitude > (self.tol*np.max(magnitude) )
        n0 = 0
        for m0 in range(M2):
            heapq.heappush(h, (-self.magnitude[n0, m0],n0,m0))

        while len(h) > 0:
            s=heapq.heappop(h)
            n,m = s[1],s[2]
            if n==1 and m < M2-1 and mask[m+1]: # North
                mask[m+1]=False
                self.phase[n, m+1]=  self.phase[n,  m] +(self.fgrad[n,  m] + self.fgrad[n, m+1])/2
                heapq.heappush(h, (-self.magnitude[n,m+1],n,m+1))
                if self.plt.verbose and self.debug_count <= 2000 :
                    self.debugInfo(n, m+1, n, m, self.phase, self.original_phase)

            if n == 1 and m > 0 and mask[m-1]: # South
                mask[m-1]=False
                self.phase[n, m-1]=  self.phase[n,  m] - (self.fgrad[n,  m] + self.fgrad[n, m-1])/2
                heapq.heappush(h, (-self.magnitude[n,m-1],n,m-1))
                if self.plt.verbose and self.debug_count <= 2000 :
                    self.debugInfo(n, m-1, n, m, self.phase, self.original_phase)

            if n==0 and mask[m]:  # East
                mask[m]=False
                self.phase[(n+1), m]=  self.phase[n,  m] + self.time_scale*(self.tgrad[n,  m] + self.tgrad[(n+1), m])/2
                heapq.heappush(h, (-self.magnitude[n+1,m], 1, m))
                if self.plt.verbose and self.debug_count <= 2000 :
                    self.debugInfo(n+1, m, n, m, self.phase, self.original_phase)

        return self.phase[0], self.fgrad[0], self.tgrad[0]


    def sigstretch(self, samples):
        '''
            modify the FFT magnitude coefficients to translate and scale the
                frequency
                parameter:
                    magnitude
                        np.array the absolute values of the FFT coefficients
                return
                    magnitude
                        np.array
        '''
        if self.freq_scale ==1:
            return samples

        newMs = np.linspace(0, samples.size, self.freq_scale*samples.size, endpoint=False)
        newsig = np.empty_like(newMs)

        if self.freq_scale < 1 :
            lowpassfir =  signal.firwin(32, .9*self.freq_scale)
            samples = np.convolve(lowpassfir, samples, mode='same')

        for m,v in enumerate(newMs):
            oldMhigh = min(samples.size-1, int(np.ceil(v)))
            oldMlow = max(0,int(np.floor(v)))
            dv = v-oldMlow
            assert oldMhigh >=0 and oldMhigh < samples.size
            assert oldMlow >=0 and oldMlow < samples.size
            newsig[m]= (1-dv)*samples[oldMlow] + dv*samples[oldMhigh]
        return newsig

    def debugInfo(self, n1, m1, n0, m0, phase, original_phase):
        dif = (phase[n1,m1] - phase[n0,m0]) %(2*np.pi)
        if original_phase is None:
            dif_orig = dif
        else:
            if n1 != n0:
                dif_orig = (original_phase[n1,m1] - original_phase[n0,m0] )%(2*np.pi)
            elif m1 != m0:
                dif_orig = (original_phase[n1,m1] - original_phase[n0,m0])%(2*np.pi)
        if dif_orig==0:
            err_new = 0
        else:
            err_new = (dif - dif_orig) /dif_orig
        self.q_errors.append((n0,m0,0 ,n1-n0,m1-m0,err_new/(2*np.pi)))

        if self.debug_count < 10:
            if m1 == m0+1:
                self.logprint('###############################   POP   ###############################')
            self.logprint(['','NORTH','SOUTH'][m1-m0]+ ['','EAST','WEST'][n1-n0])
            self.logprint ('n1,m1=({},{}) n0,m0=({},{})'.format(n1,m1,n0,m0))
            self.logprint ('\testimated phase[n,m]={:13.4f}, phase[n0,m0]         =:{:13.4f}, dif(2pi)     ={:9.4f}'.format((phase[n1,m1]) , (phase[n0,m0]), dif ))
            if original_phase is not None:
                self.logprint ('\toriginal_phase[n,m] ={:13.4f}, original_phase[n0,m0]=:{:13.4f}, dif_orig(2pi)={:9.4f}'.format((original_phase[n1,m1]) , (original_phase[n0,m0])  ,dif_orig))
                self.logprint('error ={:9.4f}%'.format(100*err_new))
        self.debug_count += 1

    def magphase_to_complex(self,magnitude, phase):
        return magnitude*(np.cos(phase)+ np.sin(phase)*1j)

    def magphase_frames_to_signal(self, magnitude, phase):
        return self.complex_frames_to_signal(self.magphase_to_complex(magnitude, phase))

    def complex_to_magphase(self, corig ):
        return  np.absolute(corig),np.angle(corig)

    def signal_to_frames(self, s):   # applies window function, g
        self.plt.signal_to_file(s , 'signal_in' )
        self.plt.spectrogram(s,'spectrogram signal in')
        L = s.shape[0] - self.M
        self.corig_frames = np.stack( [np.fft.rfft(self.g*s[ix:ix + self.M]) for ix in range(0, L, self.a_a)])
        return self.corig_frames

    def complex_frames_to_signal(self, complex_frames):
        M2 = complex_frames.shape[1]
        N = complex_frames.shape[0]
        M = self.M
        a_s = self.a_s

        vr=np.fft.irfft(complex_frames)
        sig = np.zeros((N*a_s+self.M))
        cum_waveforms=[]
        n1 = 15
        n2 = 25
        for n in range(N):
            vs = vr[n]*self.gsynth
            if self.verbose and n >= n1 and n < n2:
                vout = np.zeros(((n2-n1)*a_s+M))
                na = (n-n1)*a_s
                vout[na:na+M] = vs
                cum_waveforms.append(vout)
            sig[n*a_s: n*a_s+M] += vs
        self.plt.plot_waveforms('Gabor Contributions', cum_waveforms)
        self.plt.signal_to_file(sig , 'signal_out')
        self.plt.spectrogram(sig, 'spectrogram signal out')
        return sig

    def signal_to_magphase_frames(self, s):
        return self.complex_to_magphase(self.signal_to_frames(s))

    def signal_to_signal(self,signal_in):
        '''
          convert signal_in to frames
            throw away the phase
            reconstruct the phase from the magnitudes
            re-run fft and compute the frobenius norm for an error value

            parameter:
                signal_in numpy array (length,)

            return:
                 reconstructed signal
        '''
        self.plt.signal_to_file(signal_in , 'signal_in_before_stretch' )
        self.plt.spectrogram(signal_in,'spectrogram signal_in_before_stretch in')
        s= self.sigstretch(signal_in)
        magnitude_frames, _ = self.signal_to_magphase_frames(s)
        phase_estimated_frames = self.magnitude_to_phase_estimate(magnitude_frames)

        signal_out = self.magphase_frames_to_signal(magnitude_frames, phase_estimated_frames)
        self.plt.plot_waveforms('Signal in, Signal out', [signal_in, signal_out])

        saved_verbose = self.setverbose(False)
        reconstructed_magnitude, _ = self.signal_to_magphase_frames(signal_out)
        self.setverbose(saved_verbose)
        if magnitude_frames.shape[0]>1 and reconstructed_magnitude.shape[0] >1:
            s1 = self.plt.normalize(magnitude_frames[1:]) # s1 is delayed by 1 frame with respect to s2
            s2 = self.plt.normalize(reconstructed_magnitude[:-1])
            minlen = min(s1.shape[0], s2.shape[0])
            s1 = s1[:minlen]
            s2 = s2[:minlen]
            mn = min(minlen,100)-15
            dif = s2 - s1
            E = np.sqrt(np.sum(dif*dif)) / np.sqrt(np.sum(s1*s1))   # Frobenius norm
            self.plt.plot_3d('magnitude_frames, reconstructed_magnitude', [s1[mn:mn+10], s2[mn:mn+10] ])
        return signal_out










