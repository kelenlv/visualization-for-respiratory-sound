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
    plt.grid(ls="--")
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

#def update_points(num):
#    '''
#    更新数据点
#    '''
#    point_ani.set_data(x[num], y[num])
#    return point_ani,
#
#x = np.linspace(0, 2*np.pi, 100)
#y = np.sin(x)
#
#fig = plt.figure(tight_layout=True)
#plt.plot(x,y)
#point_ani, = plt.plot(x[0], y[0], "ro")
#plt.grid(ls="--")
## 开始制作动画
#ani = FuncAnimation(fig, update_points, np.arange(0, 100), interval=100, blit=True)
#
## ani.save('sin_test2.gif', writer='imagemagick', fps=10)
#plt.show()