# Visualization for Respiratory Sound
THe respiratory sound database is obtained from [ICBHI 2017 Challenge](https://bhichallenge.med.auth.gr/), which focuses on the classification task including four types as crackles, wheezes, a combination of them, or no adventitious respiratory sounds. 

This repo is mainly adapted from [sed_vis - Visualization toolbox for Sound Event Detection](https://github.com/TUT-ARG/sed_vis#sed_vis---visualization-toolbox-for-sound-event-detection) which is a open source toolbox. Based on it, we modified some functions for the application of visualization for the respiratoty sound, including:

 - Format transformation of input reference.
 - Redefinition of the spectrogram and new definition of the Log-Power spectrogram. 
 - New panel with  the Log-Power spectrogram. 

___
## usage:
place  modified sed_vis.egg-info in Anaconda/Lib/site-packages/, and run tests/test_sed_visualizer.sh
___
1. 整个界面：
- 可以用户自定义每个面板的X，Y轴数值范围
- 可以框选ROI并放大或缩小，且可以返回操作
- 可以开始、暂停、重置播放，及关闭界面
2. 呼吸音的增强、减弱变化
- Waveform in time domain面板
3. 呼吸音强度深浅变化
- Intensity面板+colorbar
4. 呼吸音高调低调变化，实时显示
- Real-time spectrogram 面板
5. 统计指标及展示
- Event roll 面板
- 每分钟呼吸次数（呼吸频率）
- 异常呼吸音的次数和长度
- Crackle, wheeze的呼吸音cycle数量，及时间占比
- 强度最大值（命令行显示）
