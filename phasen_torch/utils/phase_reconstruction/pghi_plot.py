'''
Created on Jul 26, 2018

@author: richard
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as signal
import numpy as np
import os
import glob
from pydub import AudioSegment
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

PLOT_POINTS_LIMIT = 20000
PLOT_TICKS_LIMIT = 5000

file_sep = ' '
class Pghi_Plot(object):
    '''
    classdocs
    '''

    def __init__(self, show_plots=True, show_frames = 5, pre_title='', soundout = './soundout/', plotdir='./pghi_plots/', Fs=44100, verbose=True, logfile='log.txt'):
        '''
        parameters:
            show_plots
                if True, then display each plot on the screen before saving
                to the disk. Useful for rotating 3D plots with the mouse
                if False, just save the plot to the disk in the './pghi_plots' directory
            pre_title
                string: pre_titleription to be prepended to each plot title
        '''
        
        self.show_plots,self.show_frames,self.pre_title,self.soundout,self.plotdir,self.Fs,self.verbose,self.logfile = show_plots,show_frames,pre_title,soundout,plotdir,Fs,verbose, logfile
        self.colors =  ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
        try:
            os.mkdir(plotdir)    
        except:
            pass
        try:
            os.mkdir(soundout)            
        except:
            pass                
        self.openfile = ''
        self.mp3List = glob.glob('./*.mp3',recursive=False) + glob.glob('./*.wav',recursive=False)
        self.fileCount=0       
        self.logprint('logfile={}'.format(logfile))   
        
    def save_plots(self, title):  
        file =  self.plotdir + title +  '.png' 
        print ('saving plot to file: ' + file)        
        plt.savefig(file,  dpi=300)    
        if self.show_plots:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()            
            plt.show()             
        else:               
            plt.clf() # savefig does not clear the figure like show does
            plt.cla()      
            plt.close()
            
    def colorgram(self, title, samples, mask=None, startpoints = None):
        if not self.verbose: return        
        if mask is not None:
            samples = samples*mask
        samples = np.transpose(samples)
        title = self.pre_title  +file_sep+title     
         
        fig = plt.figure()  
        plt.title( title )          
        ax = plt.gca()
        plt.imshow(samples, origin = 'lower', cmap='hot_r') 
        plt.xlabel('frames')
        plt.ylabel('Frequency Bins')
        plt.grid()
        self.save_plots(title)
         
    def spectrogram(self, samples, title):
        if not self.verbose: return        
        title = self.pre_title  +file_sep+title      
        plt.title( title )           
        ff, tt, Sxx = signal.spectrogram(samples,  nfft=8192)
    
        plt.pcolormesh(tt, ff, Sxx, cmap='hot_r')
        plt.xlabel('samples')
        plt.ylabel('Frequency (Hz)')
        plt.grid()
        self.save_plots(title)
        
    prop_cycle = plt.rcParams['axes.prop_cycle']    

    def plot_waveforms(self, title, sigs,fontsize=None):  
        if not self.verbose: return   
        title = self.pre_title  + file_sep + title  
        fig = plt.figure()
        plt.title(title)   
        plt.ylabel('amplitude', color='b',fontsize=fontsize)
        plt.xlabel('Samples',fontsize=fontsize)    
        ax = plt.gca()
    
        for i,s in enumerate(sigs):
            s = s[:PLOT_TICKS_LIMIT]                       
            xs = np.arange(s.shape[0])
            ys = s
            ax.scatter(xs, ys, color = self.colors[i%len(self.colors)],s=3)      
        plt.grid()
        plt.axis('tight')
        self.save_plots(title)
        
    def minmax(self, startpoints, stime, sfreq):
        ''' 
        limit the display to the region of the startpoints
        '''
        if startpoints is None:
            minfreq = mintime = 0
            maxfreq = maxtime = 2*self.show_frames
        else:
            starttimes = [s[0] for s in startpoints]
            startfreqs = [s[1] for s in startpoints]           
            
#             starttimes = [startpoints[0][0]]
#             startfreqs = [startpoints[0][1]]                       
                 
            mintime = max(0,min(starttimes)-self.show_frames)
            maxtime = min(stime,max(starttimes)+self.show_frames)
            minfreq = max(0,min(startfreqs)-self.show_frames)
            maxfreq = min(sfreq,max(startfreqs)+self.show_frames)  
            
        return mintime, maxtime, minfreq, maxfreq
    
    def subplot(self, figax, sigs, r, c, p, elev, azim, mask, startpoints, fontsize=None):
        ax = figax.add_subplot(r,c,p, projection='3d',elev = elev, azim=azim)     
        for i, s in enumerate(sigs):

            mintime, maxtime, minfreq, maxfreq = self.minmax(startpoints, s.shape[0], s.shape[1])            
            values = s[mintime:maxtime, minfreq:maxfreq]     
            values = self.limit(values)                  
            if mask is None:  #plot all values                                      
                xs = np.arange(values.size) % values.shape[0]
                ys = np.arange(values.size) // values.shape[1]
                zs = np.reshape(values,(values.size))
            else:                         
                indices = np.where(self.limit(mask[mintime:maxtime, minfreq:maxfreq])   == True)
                xs = indices[0] + mintime 
                ys = indices[1] + minfreq      
                zs = values[indices]  
            if i==0:
                sn=8
            else:
                sn=3  
            ax.scatter(xs, ys, zs, s=sn, color = self.colors[(i+1)%len(self.colors)]) 
        if xs.shape[0] > 0:
            mint = min(xs)
            maxt = max(xs)
            minf = min(ys)
            maxf = max(ys)
            if startpoints is not None:
                for stpt in startpoints:                
                    n = stpt[0] 
                    m = stpt[1]
                    if n >= mint and n <= maxt and m >= minf and m <= maxf:
                        ax.scatter([n ],[m ], [s[n,m]], s=30, color = self.colors[0])
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.zaxis.set_major_formatter(StrMethodFormatter('{x:.2e}'))            
                       
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 
            tick.label.set_rotation('vertical')
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 
            tick.label.set_rotation('vertical') 
        for tick in ax.zaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 
            tick.label.set_rotation('horizontal')    
                                                             
        ax.set_zlabel('mag',fontsize=fontsize)   
        ax.set_ylabel('STFT bin',fontsize=fontsize)
        ax.set_xlabel('frame',fontsize=fontsize)
        
    def normalize(self, mono):
        ''' return range (-1,1) '''
        return 2*(mono - np.min(mono))/np.ptp(mono) -1.0
        
    def signal_to_file(self, sig, title, override_verbose = False):  
#         print (np.max(sig),np.min(sig) ) 
        ''' stores the signal in a tile, with title   
        
        parameters
            sig 
                either an numpy array of shape (c,n)  containing the right
                and left channels, where c is the number of channels
                or a numpy array (n)which is store to the plot_files directory
            title 
                string to name the file. 
            '''
    
        if override_verbose == False:
            if not self.verbose: return     
            filename = self.plotdir+ self.pre_title  + file_sep+ title +'.mp3'   
        else:
            filename = self.soundout+'_' +title  +'.mp3'                      
            
        if len(sig.shape) == 1:
            sig = np.reshape(sig, (1,-1))
        channels=sig.shape[0]
        print('saving signal to file: {}'.format(filename))
        sig = (self.normalize(sig))*(2**15-1)    
        if np.max(sig) >= 2**15:
            print (np.argmax(sig), np.max(sig))
        if np.min(sig) < -2**15 :
            print (np.argmin(sig), np.min(sig))              
        sig = np.array(sig, dtype=np.int16)        
        sig = np.rollaxis(sig, 1)
        sig = sig.flatten()
        sig = sig[: 4*(sig.shape[0]//4)]
        output_sound = AudioSegment(data=sig, sample_width=2,frame_rate=self.Fs, channels=channels)    
        output_sound.export(filename, format="mp3")  
                        
    def plot_3d(self, title, sigs, mask=None, startpoints=None):
        if not self.verbose: return        
        title = self.pre_title  + file_sep + title       
        figax = plt.figure()   
        plt.axis('off')    
        plt.title( title ) 

        if self.show_plots:      
            self.subplot(figax, sigs, 1,1, 1, 45, 45, mask,startpoints,fontsize=8)   
        else:         
            self.subplot(figax, sigs, 2,2, 1, 45, 45, mask,startpoints,fontsize=6)
            self.subplot(figax, sigs, 2,2, 2, 0,  0,  mask,startpoints,fontsize=6)
            self.subplot(figax, sigs, 2,2, 3, 0,  45, mask,startpoints,fontsize=6)
            self.subplot(figax, sigs, 2,2, 4, 0,  90, mask,startpoints,fontsize=6)                           
        self.save_plots(title) 
        
    def limit(self, points):   
        ''' limit the number of points plotted to speed things up
        '''
        points = np.array(points)
    
        if points.size > PLOT_POINTS_LIMIT:
            s0  = int(PLOT_POINTS_LIMIT/points[0].size)
            print ('limiting the number of plotted points') 
            points = points[:s0]
 
        return points
               
    def quiver(self, title, qtuples, mask=None, startpoints=None):   
        if not self.verbose: return
        if len(qtuples)==0: return
        title = self.pre_title + file_sep + title
        qtuples = self.limit(qtuples)
        figax = plt.figure()
        ax = figax.add_subplot(111, projection='3d',elev = 45, azim=45)
        plt.title(title)        
        stime = max([q[0] + q[3] for q in qtuples])
        sfreq = max([q[1] + q[4] for q in qtuples])
        mintime, maxtime, minfreq, maxfreq = self.minmax(startpoints, stime, sfreq)             
        x, y, z, u, v, w = [],[],[],[],[],[]
        for q in qtuples:        
            if q[0] < mintime or q[0] > maxtime or q[1] < minfreq or q[1] > maxfreq:
                continue;
            x.append(q[0])
            y.append(q[1])
            z.append(q[2])
            u.append(q[3])
            v.append(q[4])
            w.append(q[5])                        
                   
        ax.quiver(x,y,z,u,v,w,length=.5, arrow_length_ratio=.3, pivot='tail', color = self.colors[1], normalize=True)
        if startpoints is not None:
            for stpt in startpoints:                
                n = stpt[0]
                m = stpt[1]
                ax.scatter([n],[m], [z[0]], s=30, color = self.colors[0])     
        self.save_plots(title)
         
    def logprint(self, txt):   
        if self.verbose:
            if self.openfile != './pghi_plots/' + self.logfile :
                self.openfile = './pghi_plots/' + self.logfile
                self.file = open(self.openfile, mode='w')
            print(txt, file=self.file, flush=True)
        print(txt)
           
    def get_song(self):
        ''' 
            get a song and keep it in self.sound_clip
            sound_clip shape= (samples, channels) where channels = 2
            sound is normalized in the range -1 to 1
        returns 
            sound_title without the .mp3 extension
            sound
                stereo numpy array (n,samples)
                where n is the number of channels, i.e. 2 = stereo            
        '''
        if self.fileCount >= len(self.mp3List):
            return None,None
        file = self.mp3List[self.fileCount]
        self.logprint('file={}'.format(file))
        _,filename=os.path.split(file)
        
        self.fileCount +=1
        try:
            song = AudioSegment.from_mp3(file)
        except:
            self.logprint("song decoding error")
            return self.get_song() # try to get next song

        if song.frame_rate != self.Fs:
            self.Fs = song.frame_rate
            self.logprint("changing frame rate")

        samples = song.get_array_of_samples()
        samples = np.array(samples,dtype=np.float32)
        samples= np.reshape(samples,(-1,song.channels))      
        samples = np.rollaxis(samples,1)     
        samples = self.normalize(samples)
        return filename.split('.')[0], samples

                   