# -*- coding: utf-8 -*-
# Script for functions that may be used in data processing

__author__ = 'Ruien Wang'


import os
import glob
import mne
from mne.viz import plot_topomap
from mne.time_frequency import psd_multitaper
from mne.stats import fdr_correction
import neurokit2 as nk
import pandas as pd
import numpy as np
import mantel
from scipy.stats import pearsonr
from scipy.io import loadmat
from scipy.stats.mstats_basic import rankdata
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from nltools.stats import isc, isps
from scipy.stats import rankdata


## Get the vector of the lower triangular marix

def get_tril_vec(matrix):
    vec = []
    for i in range(0, matrix.shape[0]-1):
        tp = matrix[i, i+1:]
        vec.extend(tp)
    vec = np.array(vec)
    return vec
## Convert ISC matrix to csv file
def eeg_isc2csv(isc_dict, n, filename):
    results_dir = 'F:/1_Emotion_Data/Results/0_ISC/ISC_CSV/'
    ch_idx = list(range(63))
    result = np.zeros((n, 63)) 
    for ch in ch_idx:
        tp = get_tril_vec(isc_dict[ch])
        result[:,ch] = tp
    pd.DataFrame(result, dtype='float64').to_csv(os.path.join(results_dir + filename + '_isc.csv'))

###  Part1 :Function for raw data import
## EEG import
def import_eeg(video,emotion_dir):
    montage = mne.channels.read_custom_montage('F:/1_Emotion_Data/Data/EEG/Emotion.loc')
    sub_list = [os.path.basename(x).split('_')[1] for x in glob.glob(os.path.join(emotion_dir, '*set'))]
    sub_list = list(map(int, sub_list))
    # Store all subjects into a dict
    eeg_meta = {}
    for sub in sub_list:
        if sub <= 9:
            file = emotion_dir + 'sub_00' + str(sub) + '_'+ video + '.set'
        else:
            file = emotion_dir + 'sub_0' + str(sub) + '_'+ video + '.set'
        # Use the mne.read_epochs_eeglab() to read the preprocessed data
        tp = mne.read_epochs_eeglab(file)
        # Set the montage
        tp.set_montage(montage)
        eeg_meta[sub] = tp
        del tp
    return sub_list, eeg_meta

## ECG import
def import_ecg(sub_list, emotion, video, dir):
    sub_list = sub_list
    # Store all subjects into a dict
    if emotion == 'angry':
        emo = 'ag'
    if emotion == 'anxiety':
        emo = 'ax'
    if emotion == 'fear':
        emo = 'fe'
    if emotion == 'helpless':
        emo = 'hl'
    if emotion == 'happy':
        emo = 'ha'
    
    ecg_meta = {}
    for sub in sub_list:
        if sub <= 9:
            file = dir + 'sub_00' + str(sub) + '_'+ str(emotion) + '_' + str(video) + '.mat'
        else:
            file = dir + 'sub_0' + str(sub) + '_'+ str(emotion) + '_' + str(video) + '.mat'
        # Use the mne.read_epochs_eeglab() to read the preprocessed data
        tp = loadmat(file)
        data = 'ecg' + '_' + str(emo) + str(video)
        ecg = tp[data]
        ecg = np.ndarray.flatten(ecg)
        ecg_meta[sub] = ecg
    return ecg_meta

## PPG import
def import_ppg(sub_list, emotion, video, dir):
    sub_list = sub_list
    # Store all subjects into a dict
    if emotion == 'angry':
        emo = 'ag'
    if emotion == 'anxiety':
        emo = 'ax'
    if emotion == 'fear':
        emo = 'fe'
    if emotion == 'helpless':
        emo = 'hl'
    if emotion == 'happy':
        emo = 'ha'
    
    ppg_meta = {}
    for sub in sub_list:
        if sub <= 9:
            file = dir + 'sub_00' + str(sub) + '_'+ str(emotion) + '_' + str(video) + '.mat'
        else:
            file = dir + 'sub_0' + str(sub) + '_'+ str(emotion) + '_' + str(video) + '.mat'
        # Use the mne.read_epochs_eeglab() to read the preprocessed data
        tp = loadmat(file)
        data = 'ppg' + '_' + str(emo) + str(video)
        ppg = tp[data]
        ppg = np.ndarray.flatten(ppg)
        ppg_meta[sub] = ppg
    return ppg_meta

## EDA import
def import_eda(sub_list, emotion, video, dir):
    sub_list = sub_list
    # Store all subjects into a dict
    if emotion == 'angry':
        emo = 'ag'
    if emotion == 'anxiety':
        emo = 'ax'
    if emotion == 'fear':
        emo = 'fe'
    if emotion == 'helpless':
        emo = 'hl'
    if emotion == 'happy':
        emo = 'ha'
    
    eda_meta = {}
    for sub in sub_list:
        if sub <= 9:
            file = dir + 'sub_00' + str(sub) + '_'+ str(emotion) + '_' + str(video) + '.mat'
        else:
            file = dir + 'sub_0' + str(sub) + '_'+ str(emotion) + '_' + str(video) + '.mat'
        # Use the mne.read_epochs_eeglab() to read the preprocessed data
        tp = loadmat(file)
        data = 'eda' + '_' + str(emo) + str(video)
        eda = tp[data]
        eda = np.ndarray.flatten(eda)
        eda_meta[sub] = eda
    return eda_meta

## EMG import
def import_emg(sub_list, emotion, video, dir):
    sub_list = sub_list
    # Store all subjects into a dict
    if emotion == 'angry':
        emo = 'ag'
    if emotion == 'anxiety':
        emo = 'ax'
    if emotion == 'fear':
        emo = 'fe'
    if emotion == 'helpless':
        emo = 'hl'
    if emotion == 'happy':
        emo = 'ha'
    
    emg_meta = {}
    for sub in sub_list:
        if sub <= 9:
            file = dir + 'sub_00' + str(sub) + '_'+ str(emotion) + '_' + str(video) + '.mat'
        else:
            file = dir + 'sub_0' + str(sub) + '_'+ str(emotion) + '_' + str(video) + '.mat'
        # Use the mne.read_epochs_eeglab() to read the preprocessed data
        tp = loadmat(file)
        data = 'emg' + '_' + str(emo) + str(video)
        emg = tp[data]
        emg = np.ndarray.flatten(emg)
        emg_meta[sub] = emg
    return emg_meta

### Part 2: Data processing

## Functon for raw signal based Whole brain ISC 
def singch_amp(df, ch):
    # Construct temporal dict for single channel amplitude
    sc_amp = {}
    for sub in df:
        sc_amp[sub] = df[sub].iloc[:,ch]
    sc_amp = pd.DataFrame(sc_amp)
    return sc_amp
        
## Function for PSD based Whole brain ISC of single channel
def singch_psd(meta_data, ch, band_min, band_max, start, end): 
    sc_psd = {}
    interval = end-start+1
    interval_start = start-1
    interval_end = end
    for sub in meta_data:
        sc_band = np.zeros((interval))
        for j in range(interval_start, interval_end):
            epoch = meta_data[sub][j]
            psds, freqs = psd_multitaper(epoch,fmin=1, fmax=50)
            psds = 10* np.log10(10**12*psds)
            delta = np.mean(np.squeeze(psds)[ch, band_min-1:band_max], axis=0)
            sc_band[j - interval_start] = delta
        sc_psd[sub] = sc_band
    sc_psd = pd.DataFrame(sc_psd)
    return sc_psd

### Function for physiological data processing
## ECG processing
def rain_ecg (ecg_signal, fs, dfs):
    # Clean the raw ECG signal
    ecg_resampled = nk.signal_resample(ecg_signal, method="numpy",sampling_rate=fs, desired_sampling_rate=dfs)
    ecg_cleaned = nk.ecg_clean(ecg_resampled, sampling_rate=dfs, method="neurokit")
    instant_peaks,rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=dfs, method="neurokit", correct_artifacts=True)
    PQRST, waves = nk.ecg_delineate(ecg_cleaned, rpeaks, sampling_rate=dfs, method="cwt", show=False)
    cardiac_phase = nk.ecg_phase(ecg_cleaned, rpeaks=rpeaks, delineate_info=waves, sampling_rate=dfs)
    rate = nk.ecg_rate(rpeaks, sampling_rate=dfs, desired_length=len(ecg_cleaned), interpolation_method='monotone_cubic')
    quality = nk.ecg_quality(ecg_cleaned, sampling_rate=dfs)
    edr = nk.ecg_rsp(rate,sampling_rate=dfs)
    # Aggregate the result
    results = pd.DataFrame({"ECG_Raw": ecg_resampled,
                            "ECG_Clean": ecg_cleaned,
                            "ECG_Rate": rate,
                            "ECG_Quality": quality,
                            "ECG_EDR":edr})

    results = pd.concat([results, instant_peaks,PQRST,cardiac_phase], axis=1)
    # ECG feature extraction(interval related in this project)
    #ecg_features=nk.ecg_intervalrelated(results,sampling_rate=fs)
    return results #ecg_features

### HRV processing
def rain_hrv(ecg_signal, sampling_rate, desireed_sampling_rate):
    ecg_resampled = nk.signal_resample(ecg_signal, method="numpy",sampling_rate=sampling_rate, desired_sampling_rate=desireed_sampling_rate)
    ecg_cleaned = nk.ecg_clean(ecg_resampled, desireed_sampling_rate, method="neurokit")
    # Find peaks
    peaks, info = nk.ecg_peaks(ecg_cleaned, desireed_sampling_rate)
    # Compute HRV indices
    result = nk.hrv_time(peaks, desireed_sampling_rate, show=False)
    return result.loc[:,['HRV_MeanNN', 'HRV_SDNN']]


## PPG Processing
def rain_ppg(ppg_signal, fs, dfs):
    # Import the raw data(1-D array), downsampling first, fs denotes the originial fs, while dfs denotes the desired fs
    ppg = nk.signal_resample(ppg_signal, method="numpy",sampling_rate=fs, desired_sampling_rate=dfs)
    # Clean the raw signal
    ppg_cleaned = nk.ppg_clean(ppg,sampling_rate=dfs)
    # Find the peaks of the PPG signal
    peak_info = nk.ppg_findpeaks(ppg_cleaned,sampling_rate=dfs,show=False)
    # Match the peak location into the timeseries
    peaks = peak_info['PPG_Peaks'] 
    peaks_signal = np.zeros(ppg.shape); peaks_signal[peaks] = 1
    # Calculate the signal rate
    rate = nk.ppg_rate(peaks, sampling_rate=dfs, desired_length=len(ppg_cleaned))
    # Aggregate the results to a dataframe
    results = pd.DataFrame({"PPG_Raw":ppg,"PPG_Clean":ppg_cleaned,"PPG_Rate":rate,"PPG_Peaks":peaks_signal})
    # Obtain the PPG features
    #ppg_features = nk.ppg_intervalrelated(signals,sampling_rate=dfs)
    return results #ppg_features

## Calculate the mean of  3 eeg dataframes
def mean_df3_eeg(sub_list, df1, df2, df3):
    meta = []
    for sub in sub_list:
        a = df1[sub].to_numpy()
        b = df2[sub].to_numpy()
        c = df3[sub].to_numpy()
        mean =[]
        for i in range(0,20):
            tp = np.array([a[i], b[i], c[i]])
            m = np.mean(tp)
            mean.append(m)
        meta.append(mean)
    meta = pd.DataFrame(meta).T
    return meta

## Calculate the mean of 2 eeg dataframes
def mean_df2_eeg(sub_list, df1, df2):
    meta = []
    for sub in sub_list:
        a = df1[sub].to_numpy()
        b = df2[sub].to_numpy()
        mean =[]
        for i in range(0,20):
            tp = np.array([a[i], b[i]])
            m = np.mean(tp)
            mean.append(m)
        meta.append(mean)
    meta = pd.DataFrame(meta).T
    return meta

## Calculate the mean of 3 physiology dataframes
def mean_df3_phy(df1, df2, df3):
    meta = []
    for i in range(df1.shape[1]):
        a = df1.iloc[:,i].to_numpy()
        b = df2.iloc[:,i].to_numpy()
        c = df3.iloc[:,i].to_numpy()
        mean =[]
        for j in range(df1.shape[0]):
            tp = np.array([a[j], b[j], c[j]])
            m = np.mean(tp)
            mean.append(m)
        meta.append(mean)
    meta = pd.DataFrame(meta).T
    return meta
## Calculate the mean of 2 physiology dataframes
def mean_df2_phy(df1, df2):
    meta = []
    for i in range(df1.shape[1]):
        a = df1.iloc[:,i].to_numpy()
        b = df2.iloc[:,i].to_numpy()
        mean =[]
        for j in range(df1.shape[0]):
            tp = np.array([a[j], b[j]])
            m = np.mean(tp)
            mean.append(m)
        meta.append(mean)
    meta = pd.DataFrame(meta).T
    return meta


def spearmanr(x, y):

    x = np.column_stack((x, y))
    x_ranked = np.apply_along_axis(rankdata, 0, x)
    
    return np.corrcoef(x_ranked, rowvar=0)[0][1]

def permutation_cor(x, y, iter=1000):

    rtest = spearmanr(x, y)

    ni = 0

    for i in range(iter):
            
        x_shuffle = np.random.permutation(x)
        y_shuffle = np.random.permutation(y)
        rperm = spearmanr(x_shuffle, y_shuffle)

        if rperm >= rtest:
            ni = ni + 1

    p = np.float64((ni+1)/(iter+1))

    return p

def behavisc_nn(n_subs, sum_score):
    behav_rank = rankdata(sum_score)# explicity convert the raw scores to ranks
    simi = np.zeros((n_subs, n_subs))
    for i in range(n_subs):
        for j in range(n_subs):
            if i < j:
                dist_ij = 1-(abs(behav_rank[i]-behav_rank[j])/n_subs) 
                simi[i,j] = dist_ij
                simi[j,i] = dist_ij
    return simi





