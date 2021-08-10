# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 17:49:51 2021

@author: lvkexin
"""
'''AxesSubplot and Line2D objects are iterable? -> return comma'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'r-', animated=False)
#print(ln.type)#Line2D
def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
#    print(ax.type)#AxesSubplot
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
plt.show()
