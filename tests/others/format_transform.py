# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 13:03:10 2021

@author: lvkexin
"""

path=r"C:\Users\lvkexin\Downloads\sed_vis-master\tests\130_2b2_Ar_mc_AKGC417L.txt"
with open(path, "r", encoding="utf-8") as f:
    file=''
    for line in f.readlines(): 
        if '\t1\t0' in line:
            line=line.replace('\t1\t0','\tcrackles')
        elif '\t0\t1' in line:
            line=line.replace('\t0\t1','\twheezes')
        elif '\t1\t1' in line:
            line=line.replace('\t1\t1','\tboth')
        elif '\t0\t0' in line:
            line=line.replace('\t0\t0','\tnormal')
        file+=line
with open(path,"w",encoding="utf-8") as f:
        f.write(file)