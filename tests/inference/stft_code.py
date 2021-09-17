# -*- coding: utf-8 -*-
import scipy.io.wavfile as wav
import os
import shutil
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display as display
from pydub import AudioSegment
from scipy.signal import butter, lfilter  
def check_fs(dir):
    """check the fe of raw record"""

    for file in os.listdir(dir):
        fs,sig,bits= wav.read(dir+file)
        print(file,fs)

def clip_test(dir):
    """seprate trainset and testset"""

    txt_dir = dir+'train_test.txt'
    
    with open(txt_dir,'r') as f:
        name = []
        set_type = []
        for row in f.readlines():
            row = row.strip('\n')
            row = row.split('\t')
            
            name.append(row[0])
            set_type.append(row[1])
    
    for i in range(len(name)):
        if set_type[i]=='test':
            shutil.move(dir+'ICBHI_final_database/'+name[i]+'.wav', dir+'testset/'+name[i]+'.wav') 
            

            
def Normalization(x):
    x = x.astype(float)
    max_x = max(x)
    min_x = min(x)
    for i in range(len(x)):
        
        x[i] = float(x[i]-min_x)/(max_x-min_x)
           
    return x

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
 
 
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def save_pic(wav_dir,save_dir):
    txt_name = ''
    txt_dir ='../tests/'
    #wav_dir=""
    for file in os.listdir(wav_dir):
        num = file[-5]
        if file[:22]!=txt_name[:-4]:
            txt_name = file[:22]+'.txt'
            array = np.loadtxt(txt_dir+txt_name)
            #label = array[:,2:4]
            
        fs,sig= wav.read(wav_dir+'/'+file)
        
        sig = Normalization(sig)

        if fs>4000:
            sig = butter_bandpass_filter(sig, 1, 4000, fs, order=3)
        stft = librosa.stft(sig, n_fft=int(0.02*fs), hop_length=int(0.01*fs), window='hann')
        
        if fs>4000:
            display.specshow(librosa.amplitude_to_db(stft[0:int(len(stft)/2),:],ref=np.max),y_axis='log',x_axis='time')
        else:
            display.specshow(librosa.amplitude_to_db(stft,ref=np.max),y_axis='log',x_axis='time')

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)

        plt.savefig(save_dir+file[:24]+'png', cmap='Greys_r')


        plt.close()
       
        
def main(file):

   #save_pic(,'/opt/disk1/HuangYiZhang/lungsound/analysis/stft/train/')
   save_pic(file,'../inference/stft/')
