# Negative-Emotion-Neurophysiogical-Sociability <img src="https://raw.githubusercontent.com/andlab-um/Emotion_Neurophysio_IS-RSA/main/demo.png" align="right" width="490px">

[![GitHub repo size](https://img.shields.io/github/languages/code-size/andlab-um/Emotion_Neurophysio_IS-RSA?color=brightgreen&label=repo%20size&logo=github)](https://github.com/andlab-um/Emotion_Neurophysio_IS-RSA)
[![Twitter URL](https://img.shields.io/twitter/url?label=%40ANDlab3&style=social&url=https%3A%2F%2Ftwitter.com%2Flizhn7)](https://twitter.com/ANDlab3)
[![Twitter URL](https://img.shields.io/twitter/url?label=%40ruien_wang&style=social&url=https%3A%2F%2Ftwitter.com%2Flizhn7)](https://twitter.com/ruien_wang)

> From [Affective, Neuroscience, and Decision-making Lab](https://andlab-um.com)



## Highlights
* We combined virtual reality & simultaneous EEG-ECG recording to probe the neurophysiological responses of nehative emotions
* We first found a common neurophysiological response pattern of negative emotions under virtual reality 
* Furthermore, individual variation of sociability could be captured by neurophysiological responses
* Virtual reality has promising utility in naturalistic neuroimaging and socio-affective research


## Description
* This repo mainly contains scripts for processing multimodal neurophysiological data from the VR emotion project. 
* Data modality: EEG (Brain Porduct 64 ch, montage see *BP_Montage*), ECG (BIOPAC 3 leads)
* Paradigm: naturalistic viewing of negative emotinal videos under virtual reality (Unity + Steam VR)
* VR googles: VIVE Pro EYE (HTC)
* Main analysis method: Intersubject similarity analysis (ISC) & Intersubject representational similarity analysis (IS-RSA)

## Details of the scripts
* 1_Emotion_Rating: visualization of the perceived emotion arousal rating score
* 2_ISC_EEG: intersubejct simialrity analysis of the EEG data
* 3_ISC_ECG: intersubject similarity analysis of the ECG signal
* 4_ISC_Behav: intersubject similarity analysis of the sociability (mentalizing & empathy)
* 5_ISRSA: intersubject representational similarity analysis 
* 6_plots_stats: plot the topographys of ISC & IS-RSA
* Unity_call_python: scripts for using Unity platform call python codes

## Requirements

Python

```bash
python 3.8
mne
neurokit2
pandas
numpy
scipy
statsmodels
matplotlib
seaborn

```
