# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 11:01:29 2021

@author: lvkexin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
 
# Prepare Data
y, sr = librosa.load(r'C:\Users\lvkexin\Downloads\sed_vis-master\tests\130_2b2_Ar_mc_AKGC417L.wav')
print(sr)
librosa.display.waveplot(y, sr=sr)
plt.show()

Xdb = librosa.stft(y)
#Xdb = librosa.amplitude_to_db(abs(X))
plt.figure()
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.figure()
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()


S = np.abs(librosa.stft(y))
#print(librosa.power_to_db(S ** 2))
# array([[-33.293, -27.32 , ..., -33.293, -33.293],
#        [-33.293, -25.723, ..., -33.293, -33.293],
#        ...,
#        [-33.293, -33.293, ..., -33.293, -33.293],
#        [-33.293, -33.293, ..., -33.293, -33.293]], dtype=float32)

plt.figure()
plt.subplot(2, 1, 1)
librosa.display.specshow(S ** 2, sr=sr, y_axis='log')  # 从波形获取功率谱图
plt.colorbar()
plt.title('Power spectrogram')
plt.subplot(2, 1, 2)
# 相对于峰值功率计算dB, 那么其他的dB都是负的，注意看后边cmp值
librosa.display.specshow(librosa.power_to_db(S ** 2, ref=np.max),
                         sr=sr, y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-Power spectrogram')
#plt.set_cmap("autumn")
plt.tight_layout()
#plt.savefig('dB.png')
plt.show()

#df = pd.read_csv("https://github.com/selva86/datasets/raw/master/economics.csv", parse_dates=['date']).head(100)
#x = np.arange(df.shape[0])
#y_returns = (df.psavert.diff().fillna(0)/df.psavert.shift(1)).fillna(0) * 100
## diff() 将本行与前一行直接做差 shift()为向后移动一行
# 
## Plot
#plt.figure(figsize=(16,10), dpi= 80)
#plt.fill_between(x[1:], y_returns[1:], 0, where=y_returns[1:] >= 0, facecolor='green', interpolate=True, alpha=0.7)
#plt.fill_between(x[1:], y_returns[1:], 0, where=y_returns[1:] <= 0, facecolor='red', interpolate=True, alpha=0.7)
# 
## Annotate
##plt.annotate('Peak \n1975', xy=(94.0, 21.0), xytext=(88.0, 28),
##             bbox=dict(boxstyle='square', fc='firebrick'),
##             arrowprops=dict(facecolor='steelblue', shrink=0.05), fontsize=15, color='white')
## 
# 
## Decorations
#xtickvals = [str(m)[:3].upper()+"-"+str(y) for y,m in zip(df.date.dt.year, df.date.dt.month_name())]
#plt.gca().set_xticks(x[::6])
#plt.gca().set_xticklabels(xtickvals[::6], rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'})
#plt.ylim(-35,35)
#plt.xlim(1,100)
#plt.title("Month Economics Return %", fontsize=22)
#plt.ylabel('Monthly returns %')
#plt.grid(alpha=0.5)
#plt.show()
