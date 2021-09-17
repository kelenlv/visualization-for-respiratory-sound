import scipy.io.wavfile as wav
import os
import shutil
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display as display
from pydub import AudioSegment
from scipy.signal import butter, lfilter  

def clip_cycle(file,new_dir):
    """clip the record into breath cycle
    dir : trainset/testset record path
    new_dir:breath cycle save path
    """
    #for file in os.listdir(dir):
    txt_name = './'+file[:-4]+'.txt'
    #print (np.loadtxt(txt_name))
    time = np.loadtxt(txt_name)[:,0:2]
    #print (time)
    sound = AudioSegment.from_wav(file)
    for i in range(time.shape[0]):
        start_time = time[i,0]*1000
        stop_time = time[i,1]*1000
        word = sound[start_time:stop_time]
        word.export(new_dir+file.split("/")[-1][:-4]+str(i)+'.wav', format="wav")


#clip_cycle("/opt/disk1/HuangYiZhang/lungsound/testset/","/opt/disk1/HuangYiZhang/lungsound/new_testset/")