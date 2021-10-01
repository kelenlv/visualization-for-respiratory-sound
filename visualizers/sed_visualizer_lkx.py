#!/usr/bin/env python
"""
Visualizer for sound event detection system
"""

import sys
import os
import argparse
import textwrap
#import sed_vis
import dcase_util
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
import librosa
import numpy
sys.path.append('..')
import sed_vis
from inference import inference
#__version_info__ = ('0', '1', '0')
#__version__ = '.'.join(__version_info__)
def cal_fre(path,n_add):
    data,s=librosa.load(str(path),sr=None)
   
    print(data.shape)
    for i in range(len(n_add)):
        temp = n_add[i]
#        print(temp)
#        print(float(temp[0])*10)
        d = data[int(float(temp[0])*44100):int(float(temp[1])*44100)]
        mag = np.abs(librosa.stft(d,n_fft=2048))#D(f,t) shape:1025*1723
        print(mag.shape)
        print(np.median(mag))
        print(mag[0].shape)

#    
#    return median
def cal_stat(l,pred_class):
    path=l[0]
#    print(path)
    with open(path, "r", encoding="utf-8") as f: 
        crackles=[]
        t_cra=[]
        wheezes=[]
        t_whe=[]
        both=[]
        t_both=[]
        normal=[]
        t_nor=[]
        n_add=[]
        nline=0
        data=f.readlines()
#        print(data)
        for idx,line in enumerate(data): 
            #print (line)
            line+=("   "+pred_class[idx])
            if 'crackles' in line:
                a=re.findall(r'\d+\.?\d*',line)
                t=re.findall(r'-?\d+\.?\d*e?-?\d*?',line)
                crackles.append(float(a[1])-float(a[0]))
                t_cra.append(float(t[1])-float(t[0]))
            elif 'wheezes' in line:
                a=re.findall(r'\d+\.?\d*',line)
                wheezes.append(float(a[1])-float(a[0]))
                t=re.findall(r'-?\d+\.?\d*e?-?\d*?',line)
                t_whe.append(float(t[1])-float(t[0]))
            elif 'both' in line:
                a=re.findall(r'\d+\.?\d*',line)
                both.append(float(a[1])-float(a[0]))
                t=re.findall(r'-?\d+\.?\d*e?-?\d*?',line)
                t_both.append(float(t[1])-float(t[0]))
            elif 'normal' in line:
                a=re.findall(r'\d+\.?\d*',line)
                normal.append(float(a[1])-float(a[0]))
                t=re.findall(r'-?\d+\.?\d*e?-?\d*?',line)
                n_add.append(t)
#                print(t)
                t_nor.append(float(t[1])-float(t[0]))
            nline+=1  
        tt=sum(t_both)+sum(t_nor)+sum(t_cra)+sum(t_whe)
    
    data = {" ": ['normal','crackles', 'wheezes', 'both'],
    "number": [len(normal), len(crackles), len(wheezes),len(both)],
     "total time (s)": [round(sum(t_nor),2), round(sum(t_cra),2), round(sum(t_whe),2),round(sum(t_both),2)],
     "mean time (s)": [round(np.mean(t_nor),2),round(np.mean(t_cra),2),round(np.mean(t_whe),2),round(np.mean(t_both),2)],
     "time percentage (%)": [round(sum(t_nor)/tt*100,2),round(sum(t_cra)/tt*100,2),round(sum(t_whe)/tt*100,2),round(sum(t_both)/tt*100,2)]}

    df = pd.DataFrame(data)
    return df,n_add

def process_arguments(argv):

    # Argparse function to get the program parameters
    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            Sound Event Visualizer
        '''))

    # Setup argument handling
    parser.add_argument('-a',
                        dest='audio_file',
                        default=None,
                        type=str,
                        action='store',
                        help='<Required> Audio file',
                        required=True)

    parser.add_argument('-l',
                        '--list',
                        nargs='+',
                        help='<Required> List of event list files',
                        required=True)

    parser.add_argument('-n',
                        '--names',
                        nargs='+',
                        help='List of names for event lists files (same order than event list files)',
                        required=False)

    parser.add_argument('-e',
                        '--events',
                        nargs='+',
                        help='List of active event classes',
                        required=False)

    parser.add_argument('--time_domain',
                        help="Time domain visualization",
                        action="store_true")

    parser.add_argument('--spectrogram',
                        help="Spectrogram visualization <default>",
                        action="store_true")

    parser.add_argument('--minimum_event_length',
                        help="Minimum event length",
                        type=float)

    parser.add_argument('--minimum_event_gap',
                        help="Minimum event gap",
                        type=float)

    parser.add_argument('--publication',
                        help="Strip visual elements out, use to generate figures for publication",
                        action="store_true")

#    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
    
    parser.add_argument('-sp',
                        dest='save_path',
                        default=None,
                        type=str,
                        help="Save the figure at the given path without opening a figure window (useful for batch "
                             "processing of the figures",
                        action='store',
                        required=False)
    
    return vars(parser.parse_args(argv[1:]))


def write_class_to_txt(pred_class,file):
    file_data = ""
    #print (file)
    with open(file[0], "r") as f:
        for idx,line in enumerate(f):
             line=line.strip() 
             #print (line)
             line=line.split("\t")
             print (line)
             file_data += line[0]+"\t"
             file_data += line[1]+"\t"
             file_data += pred_class[idx]
             file_data += "\n"
    print (file_data)
    with open(file[0],"w") as f:
        f.write(file_data)

def main(argv):
    """
    """
#    ui = dcase_util.ui.FancyPrinter()
#    ui.section_header('sed_visualizer')
    parameters = process_arguments(argv)
    
    pred_class=inference.main(parameters["audio_file"])
    #print (pred_class)
    write_class_to_txt(pred_class,parameters["list"])
    
    df, n_add=cal_stat(parameters['list'],pred_class)
    cal_fre(parameters['audio_file'],n_add)
    
    if parameters['spectrogram']:
        mode = 'spectrogram'

    elif parameters['time_domain']:
        mode = 'time_domain'

    else:
        mode = None

    if parameters['events']:
        active_events = parameters['events']
    else:
        active_events = None

    if parameters['publication']:
        publication_mode = True

    else:
        publication_mode = False

    audio_container = dcase_util.containers.AudioContainer().load(
        parameters['audio_file']
    )
#    ui.data(field='Audio file', value=parameters['audio_file'])
#
    event_lists = {}
    event_list_order = []

#    ui.line('Event lists', indent=2)
#    ui.row('ID', 'Label', 'Event list file',
#           widths=[5, 15, 40],
#           types=['int', 'str15', 'str80'],
#           indent=4
#           )
#    ui.row('-', '-', '-')
    for id, list_file in enumerate(parameters['list']):
#        ui.row(id, parameters['names'][id], list_file)

        event_lists[parameters['names'][id]] = dcase_util.containers.MetaDataContainer().load(list_file)
        event_list_order.append(parameters['names'][id])

#    ui.line()
#    ui.data(field='Mode', value=mode)
#    ui.data(field='Active events', value=active_events)
#    ui.data(field='Publication mode', value=publication_mode)
#    ui.data(field='minimum event length', value=parameters['minimum_event_length'], unit='sec')
#    ui.data(field='minimum event gap', value=parameters['minimum_event_gap'], unit='sec')
#    ui.sep()

    vis = sed_vis.visualization.EventListVisualizer(
        event_lists=event_lists,
        event_list_order=event_list_order,
        active_events=active_events,
        audio_signal=audio_container.data,
        sampling_rate=audio_container.fs,
        mode=mode,
        minimum_event_length=parameters['minimum_event_length'],
        minimum_event_gap=parameters['minimum_event_gap'],
        publication_mode=publication_mode,
    )
#    print(event_lists.get('reference'))
#    fig = plt.figure()
#    plt.axis([0, 10, 0, 10])
##    a= 20.2
##    t = ("a=%s",a)
#    
#    plt.text(5,5,event_lists.get('reference').to_string(show_info=False,show_data=False,show_stats=True) , fontsize=18, style='oblique', ha='center',va='center_baseline',wrap=True)
#    
#    plt.show()
    
#    event_lists.get('reference').to_string(show_info=False,show_data=False,show_stats=True)
#    print(event_lists.get('reference').to_string(show_info=False,show_data=False,show_stats=True))
#    event_lists.get('reference').log(level='info', show_data=False, show_stats=True)
#    print(event_lists.get('reference').event_stat_counts)

    
    
    if parameters['save_path'] is not None:
        vis.save(parameters['save_path'])
    else:
        vis.show(df)
    

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))

    except (ValueError, IOError) as e:
        sys.exit(e)
