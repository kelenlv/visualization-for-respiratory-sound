# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 10:50:18 2021

@author: lvkexin
"""
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

fig = plt.figure()
plt.axis([0, 10, 0, 10])
a= 20.2
t = ("a=%s",a)

plt.text(5, 10, t, fontsize=18, style='oblique', ha='center',va='top',wrap=True)

plt.show()



#data_matrix = [['Country', 'Year', 'Population'],
#               ['United States', 2000, 282200000],
#               ['Canada', 2000, 27790000],
#               ['United States', 2005, 295500000],
#               ['Canada', 2005, 32310000],
#               ['United States', 2010, 309000000],
#               ['Canada', 2010, 34000000]]
#fig=ff.create_table(data_matrix, height_constant=10)
#fig.show()




