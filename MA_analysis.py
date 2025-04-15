# Imports
import numpy as np
import numpy.matlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
from typing import NamedTuple
from scipy import signal
#from scipy import stats
import json
from datetime import datetime
from datetime import timezone
import sys
import math
from peakdetect import peakdetect
from pathlib import Path
#import shrd
import os
from glob import glob
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

# np.set_printoptions(threshold=sys.maxsize)

# Class Defines


class paramstruct(NamedTuple):
    fs: int
    total_timeoat
    # HR Parameters
    vital_w: int
    hs: tuple
    hs_len: float
    hr: tuple
    hs_thresh: tuple
    activity_thresh: tuple
    min_scale: int
    max_scale: int
    min_scale_resp: int
    max_scale_resp: int
    fs_ds: int
    resp: tuple
    ds_resp: int
    fs_resp: int
    # Cough Parameters
    fun_rng: tuple
    spec_dt: float
    spec_ovlp: float
    f1_pow: float
    f2_eps: float
    min_cough_sep: float
    cough_report_interval: float
    # Activity
    activity: tuple
    fs_activity: int
    # Body orientation
    fs_bo: int
    bo_cutoff: int
    ds_bo: int

    
def autocorr(x):
    # result = numpy.correlate(x, x, mode='full')[len(x)-1:]
    result = signal.correlate(x,x)[len(x)-1:]
    return result/result[0]


# Functions
def AMPD_pks(x, min_scale=None, max_scale=None):
    """Find peaks in quasi-periodic noisy signals using ASS-AMPD algorithm.
    AMPD_PKS Calculates the peaks of a periodic/quasi-periodic signal
    Method adapted from Scholkmann et.al. (2012)
    An Efficient Algorithm for Automatic Peak Detection in Noisy Periodic and
    Quasi-Periodic Signals"""

    x = signal.detrend(x)
    N = len(x)

    L = max_scale // 2
    cut = min_scale // 2

    # create LSM matix
    LSM = np.ones((L, N), dtype=bool)
    for k in np.arange(1, L + 1):
        # compare to right neighbours
        LSM[k - 1, 0:N - k] &= (x[0:N - k] > x[k:N])
        LSM[k - 1, k:N] &= (x[k:N] > x[0:N - k])  # compare to left neighbours

    G = LSM.sum(axis=1)
    # normalize to adjust for new edge regions
    G = G * np.arange(N // 2, N // 2 - L, -1)
    l_scale = cut+np.argmax(G[cut:])

    # find peaks that persist on all scales up to l
    pks_logical = np.min(LSM[0:l_scale, :], axis=0)
    pks = np.flatnonzero(pks_logical)

    return pks


def lengthtransform(x, w, fs):
    # LENGTHTRANSFORM Computes the length transform of signal <sig>
    # Length transform as described in Zong, Moody, Jiang (2003) A Robust
    # Open-source Algorithm to Detect Onset and Duration of QRS Complexes.
    # Length transform is simply the curve length with different windows <w>
    # resulting in output LT as a function of window length and sample
    C = 1/(fs**2)
    w_N = int(np.ceil(w*fs))
    normfactor = w_N/fs
    dy_k = np.array(np.diff(x, prepend=0))
    dL = np.sqrt(C + dy_k**2)

    LT = dL.cumsum()
    LT[w_N:] = LT[w_N:] - LT[:-w_N]
    return(LT-normfactor)


def shannon_energy_env(x):
    x_env = -x**2 * np.log(x**2)
    return(x_env)


def normalize(x):
    #mu = np.mean(x)
    #std = np.std(x)
    return(x)  # /np.abs(np.max(x)))#(x-mu)/std)

# Heart Rate Calculations


def calculate_b2b(envelope, length_transform, params):
    pks_sh = AMPD_pks(envelope, min_scale=params.min_scale,
                      max_scale=params.max_scale)
    pks_lt = AMPD_pks(length_transform,
                      min_scale=params.min_scale, max_scale=params.max_scale)
    N_detected = len(pks_sh)

    # plt.plot(x['hs_sh'])
    # plt.plot(pks_sh,x['hs_sh'].iloc[pks_sh],'r*',markersize=30)
    pks_sh = pks_sh[(envelope[pks_sh] > params.hs_thresh[0]) &
                    (envelope[pks_sh] < params.hs_thresh[1])]

    # Check agreement
    if N_detected:
        residual = np.min(pks_sh[np.newaxis, :] -
                          pks_lt[:, np.newaxis], axis=0)
        # 150ms is ANSI standard window for 1 beat
        matched = pks_sh[residual < .150*params.fs_ds]
        # matched = pks_sh

        # Find Beat to Beat Intervals
        b2b = np.diff(matched)/params.fs_ds
        # Check for accidental peaks in between real beats
        for k in range(len(b2b)-1):
            if b2b[k] < 60/params.hr[1] and b2b[k+1] < 60/params.hr[1]:
                b2b[k] = b2b[k] + b2b[k+1]
                b2b[k+1] = 0
                matched[k] = 0
        # remove intervals outside expected HR range
        b2b = b2b[(b2b < 60/params.hr[0]) & (b2b > 60/params.hr[1])]
        #matched[:-1] = matched[(b2b<60/params.hr[0]) & (b2b>60/params.hr[1])]
        matched = matched[matched > 0]
        N_cleaned = len(b2b+1)

        SQI = N_cleaned/N_detected
    else:
        matched, b2b, SQI = np.nan, np.nan, 0

    return(pks_sh, pks_lt, matched, b2b, SQI)

# Cough Counts


def talking_coughing(x, params, **kwargs):
    # Returns time points of cough and talk events
    return_spec = kwargs.get('return_spec', None)

    # Highpass filter 10Hz for movement
    sos = signal.butter(
        4, params.activity[1]/(params.fs/2), btype='high', output='sos')
    sig = signal.sosfiltfilt(sos, x)

    # Spectrogram
    nperseg, noverlap = int(np.floor(params.spec_dt*params.fs)
                            ), int(np.floor((params.spec_ovlp)*params.fs))
    f, t, Sxx = signal.spectrogram(sig, fs=params.fs, nperseg=nperseg, window=(
        'tukey', .25), noverlap=noverlap, mode='psd')

    Sxx = 20*np.log10(np.sqrt(Sxx)*nperseg)

    # talk time search
    idx_fun = (f > params.fun_rng[0]) & (f < params.fun_rng[1])
    # Search for First Fundamental
    fun1_pow = np.max(Sxx[idx_fun, :], 0)
    idx_max = np.argmax(Sxx[idx_fun, :], 0)
    ff_trunc = f[idx_fun]
    fun1_val = ff_trunc[idx_max]
    # Search for Second Harmonic
    ff = np.transpose(np.matlib.repmat(f, len(t), 1))
    f1val_rep = np.matlib.repmat(fun1_val, len(f), 1)
    idx_2fun = (ff > 3.0/2.*f1val_rep) & (ff < 7.0/2.*f1val_rep)
    idx_2pk = np.argmax(Sxx*idx_2fun, 0)
    fun2_val = f[idx_2pk]
    # Talk time index
    idx_tlk = np.argwhere((fun2_val >= (fun1_val*2-params.f2_eps)) & (fun2_val <= (
        fun1_val*2+params.f2_eps)) & (fun2_val >= 120) & (fun1_pow > params.f1_pow))

    idx_bw = f > params.activity[1]
    pow_sum = np.sum(Sxx[idx_bw, :], 0)
    #pow_sum = np.sum(Sxx,0)

    if len(idx_tlk):
        win, val = int(np.floor(.01/np.mean(np.diff(t)))), -7500
        for i in range(len(idx_tlk)):
            idx_start = int(idx_tlk[i] - win)
            idx_end = int(idx_tlk[i] + win)
            if (idx_start < 0):
                pow_sum[:idx_end] = val
            elif (idx_end > len(pow_sum)):
                pow_sum[idx_start:] = val
            else:
                pow_sum[(idx_start):(idx_end)] = val

    # cough count
    coughs, _ = signal.find_peaks(
        pow_sum, distance=params.min_cough_sep/np.mean(np.diff(t)), height=0)

    if return_spec:
        return(t[idx_tlk], t[coughs], f, t, Sxx)
    else:
        return(t[idx_tlk], t[coughs])


def calculate_activity(x, params, band):
    f, Pxx = signal.welch(x, fs=params.fs_ds, return_onesided=True)
    idx = (f > band[0]) & (f < band[1])
    return(np.sum(Pxx[idx]))


def downsample_stages(x, original, target):
    next_downsample = 1
    num_down = 0
    current = original
    # While we want to downsample
    while current / (next_downsample * 2) > target:
       # Increase the downsample factor
        next_downsample *= 2
        num_down += 1
        # If we don't want to use a greater downsample factor,
        # Do the downsample
        if next_downsample * 2 > 8:
            x = signal.decimate(x, next_downsample)
            current = current / next_downsample
            next_downsample = 1

    if (current/(next_downsample)-target) > (target-current/(next_downsample*2)):
        x = signal.decimate(x, next_downsample*2)  # Final downsample
        num_down += 1
    else:
        x = signal.decimate(x, next_downsample)  # Final downsample
    return(x, 2**num_down)


def resp_orient(a, params):
    n = a.shape

    # normalize each acceleration vector
    a = a/np.sqrt(a[0, :]**2+a[1, :]**2+a[2, :]**2)

    # calulate adjacent rotation angles and vectors
    theta = np.array([np.arccos(np.dot(a[:, i+1], a[:, i]))
                     for i in range(n[1]-1)])
    r = np.array([np.cross(a[:, i+1], a[:, i]) for i in range(n[1]-1)])

    # Calculate covariance matrix and PCA
    C = np.cov(r.T)
    eigval, eigvec = np.linalg.eig(C)
    r_ref = eigvec[:, np.argmax(eigval)]  # prevailing rotational axis

    # force rotation axis into same hemisphere (rotations back and forth would otherwise flip axis)
    r_t = np.array([np.multiply(r[:, i], np.sign(np.dot(r, r_ref)))
                   for i in range(3)])

    # Isolate predominant rotational axis at a point by weighting by instantaneous angle and nearby points
    W_norm = int(30*params.fs_resp)
    h_win = np.hamming(W_norm)
    r_t_norm = np.array([np.convolve(np.multiply(
        theta, r_t[i, :]), h_win, mode='same') for i in range(3)])
    r_t_norm = r_t_norm / \
        np.sqrt(r_t_norm[0, :]**2+r_t_norm[1, :]**2+r_t_norm[2, :]**2)

    # Average acceleration axis (presumed direction of gravity)
    a_ctrl_norm = np.array(
        [np.convolve(a[i, :], np.ones(W_norm)/W_norm, mode='same') for i in range(3)])
    a_ctrl_norm = a_ctrl_norm / \
        np.sqrt(a_ctrl_norm[0, :]**2+a_ctrl_norm[1, :]**2+a_ctrl_norm[2, :]**2)

    # calculate rotation angles about gravity axis
    phi = np.array([np.arcsin(np.dot(
        np.cross(a_ctrl_norm[:, i], r_t_norm[:, i]), a[:, i])) for i in range(n[1]-1)])

    sos_resp = signal.butter(8, [x/(params.fs_resp/2)
                             for x in params.resp], btype='bandpass', output='sos')
    phi = signal.sosfiltfilt(sos_resp, phi)

    # calculate respiration count via state machine
    #angular_rate = np.diff(phi)

    sig = phi  # angular_rate
    rms_resp = np.std(sig)

    cycle = list()
    resp_count = 0
    next_state = 4
    prev_loc = np.abs(sig[0]) <= rms_resp, sig[0] > rms_resp
    for i in range(len(sig)):
        cur_state = next_state
        loc = np.abs(sig[i]) <= rms_resp, sig[i] > rms_resp

        move = loc[0] ^ prev_loc[0]
        where = loc[1]
        if cur_state == 4:
            # entry state, no information on where you are
            if not loc[0]:
                next_state = where
        elif cur_state == 0:
            # we at bot
            if move:
                next_state = 2
        elif cur_state == 1:
            # we at top
            if move:
                next_state = 3
        elif cur_state == 2:
            # we at mid coming from bot
            if move and where:
                cycle.append(i)
                resp_count += 1
                next_state = 1
        elif cur_state == 3:
            # we at mid coming from top
            if move and not where:
                next_state = 0
        prev_loc = loc

        resp = np.diff(np.array(cycle)/params.fs_resp)
        resp = resp[(resp < 1/params.resp[0]) & (resp > 1/params.resp[1])]

    return sig, resp, np.mean(a_ctrl_norm, axis=1)


def interpolate_gaps(values, limit=None):
    """
    Fill gaps using linear interpolation, optionally only fill gaps up to a
    size of `limit`.
    """
    values = np.asarray(values)
    i = np.arange(values.size)
    valid = np.isfinite(values)
    filled = np.interp(i, i[valid], values[valid])

    if limit is not None:
        invalid = ~valid
        for n in range(1, limit+1):
            invalid[:-n] &= invalid[n:]
        filled[invalid] = np.nan

    return filled


def remove_spikes(x, fs, cut):
    x_n = x
    time = 1  # seconds to blank out on either side of event
    win, val = time*fs, 0  # blank out, value to set to
    idx_blank = np.argwhere(np.abs(x) > cut)
    for i in range(len(idx_blank)):
        idx_start = int(idx_blank[i] - win)
        idx_end = int(idx_blank[i] + win)
        if (idx_start < 0):
            x_n[:idx_end] = val
        elif (idx_end > len(x_n)):
            x_n[idx_start:] = val
        else:
            x_n[(idx_start):(idx_end)] = val
    return x_n


def gen_delete_list(diff):
    delete_list = []
    idx = 0
    while idx < len(diff):
        if abs(diff[idx]) > 4000:
            head = idx + 1
            tail = min(idx + 332, len(diff))
            delete_list += range(head, tail)
            idx = tail
        idx += 1

    return delete_list


def clean_data(accel_x, accel_y, accel_z):

    x_diff = np.diff(accel_x)
    y_diff = np.diff(accel_y)
    z_diff = np.diff(accel_z)

    x_delete = gen_delete_list(x_diff)
    y_delete = gen_delete_list(y_diff)
    z_delete = gen_delete_list(z_diff)
    total_delete = list(set(x_delete + y_delete + z_delete))

    accel_x = np.delete(accel_x, total_delete)
    accel_y = np.delete(accel_y, total_delete)
    accel_z = np.delete(accel_z, total_delete)

    return accel_x, accel_y, accel_z

def calculate_vitals(params, accel_x, accel_y, accel_z):
    accel_motion = savgol_filter(accel_z, 21, 8)
    accel_z_clean = accel_z - accel_motion
    accel_time = np.arange(len(accel_z_clean))/params.fs
    # accel_x = (accel.iloc[:, 4]/scale).dropna()
    # accel_y = (accel.iloc[:, 5]/scale).dropna()
    # accel_z = (accel.iloc[:, 6]/scale).dropna()
    
    # Set up HR Calculation Parameters
    params.vital_w = 10 # vital sign calculation window
    params.vital_ovlp = 0.1 # overlap of windows
    params.hs = (25, 390) # heart sounds bandpass in Hz
    params.hs_len = .165 # max length (sec) of heart sound (Luisada, Mendoza, Alimurung (1948))
    params.hs_thresh = (.0001,.3) #mouse
    params.hr = (400, 900) # mouse
    params.activity = (1, 10.) # range in Hz where power sum will be used to represent PA
    params.activity_thresh = (.00001,.5)
    params.downsample = 400 # frequency to downsample sensor data for vitals analysis
    params.resp = (1, 2.5)
    params.ds_resp = 50.
    params.bo_cutoff = 1. #low pass cutoff
    params.ds_bo = 5. # downsample frequency

    # Set up Cough Count Parameters
    params.spec_dt = 0.2 # spectrogram segment window (s)
    params.spec_ovlp = params.spec_dt-0.01 # spectrogram window overlap (s)
    params.f1_pow = 0 # minimum power (dB) for vocal fundamental frequency
    params.f2_eps = 10 # frequency +/- to search for second harmonic at 2*f1 +/- eps
    params.min_cough_sep = 1 # minimum time between coughing events (s)
    params.cough_report_interval = 60 # report count number in time in seconds

    print("PREPARED PARAMS FOR ANALYSIS")

    # Pre-process, filter data and calculate transforms
    sos_hs = signal.butter(8,np.array(params.hs)/(params.fs/2),btype='bandpass',output='sos')
    sos_shan = signal.butter(8,14/(params.fs/2),btype='low',output='sos')

    hs = signal.sosfiltfilt(sos_hs,accel_z_clean)
    hs_lt = normalize(lengthtransform(hs,params.hs_len,params.fs))
    hs_sh = normalize(signal.sosfiltfilt(sos_shan,shannon_energy_env(hs)))
    mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z_clean**2)


    print("FINISHED PARAMETERIZED PREPROCESSING")

    # Downsample using 3-stage decimation
    hs_lt,ds_hs_lt = downsample_stages(hs_lt,params.fs,params.downsample)
    hs_sh,ds_hs_sh = downsample_stages(hs_sh,params.fs,params.downsample)
    mag,ds_factor_mag = downsample_stages(mag,params.fs,params.downsample)
    #ds_factor = int(2**np.round(np.log2(params.fs/params.downsample)))
    params.fs_ds =  params.fs/ds_hs_sh
    sos_activity = signal.butter(8,[x/(params.fs_ds/2) for x in params.activity],btype='bandpass',output='sos')
    mag = signal.sosfiltfilt(sos_activity,mag)



    sos_bo = signal.butter(8,params.bo_cutoff/(params.fs/2),btype='low',output='sos')
    z_bo = signal.sosfiltfilt(sos_bo,accel_z_clean)
    body_orientation, ds_f_bo = downsample_stages(z_bo,params.fs,params.ds_bo)
    params.fs_bo = params.fs/ds_f_bo
    body_orientation [body_orientation > 1] = 1
    body_orientation [body_orientation < -1] = -1
    theta = np.arccos(body_orientation)*180/np.pi   



    ##        z_resp,ds_f_resp = downsample_stages(z_resp,params.fs,params.ds_resp)
    ##        params.fs_resp =  params.fs/ds_f_resp

    xr,_ = downsample_stages(accel_x,params.fs,params.ds_resp)
    yr,_ = downsample_stages(accel_y,params.fs,params.ds_resp)
    zr,ds_f_resp = downsample_stages(accel_z_clean,params.fs,params.ds_resp)

    params.fs_resp =  params.fs/ds_f_resp

    sos_resp = signal.butter(8,[x/(params.fs_resp/2) for x in params.resp],btype='bandpass',output='sos')
    z_resp = signal.sosfiltfilt(sos_resp,zr)
    x_resp = signal.sosfiltfilt(sos_resp,xr)
    y_resp = signal.sosfiltfilt(sos_resp,yr)
    # pd.concat([pd.Series(x_resp), pd.Series(y_resp), pd.Series(z_resp)], keys = [ 'accel_x', 'accel_y', 'accel_z'], axis=1).to_csv(path[:-8]+'_filtered_rr.csv')


    a = np.zeros((3,len(xr)))
    a[0,:] = xr
    a[1,:] = yr
    a[2,:] = zr


    print("FINISHED DOWNSAMPLING")

    # Setup Output DataFrame
    win_N = params.vital_w*params.fs_ds
    idx_v = np.arange(0,int(len(hs_sh)-win_N),int(np.floor(win_N*params.vital_ovlp)))
    vitals = np.zeros((len(idx_v),10))
    vitals[:,0] = (idx_v + win_N/2)/params.fs_ds+params.start_time    
    win_N_resp = params.vital_w*params.fs_resp
    idx_resp = np.arange(0,int(len(zr)-win_N_resp),int(np.floor(win_N_resp*params.vital_ovlp)))
    win_N_bo = params.vital_w*params.fs_bo
    idx_bo = np.arange(0,int(len(theta)-win_N_bo),int(np.floor(win_N_bo*params.vital_ovlp)))

    all_matched = []
    all_matched_amp = []
    all_pks_sh = []
    all_pks_lt = []
    all_pks_resp = []

    # Initialize starting search variables to capture correct signal features
    params.min_scale = int(np.floor(60/params.hr[1]*params.fs_ds)) # Assuming initial heart rate is non-tachycardia
    params.max_scale = int(np.ceil(60/params.hr[0]*params.fs_ds))

    # Loop through vitals calculations (HR/PA)
    beats = np.array([])
    wave = np.array([])
    #pos = np.array([])
    for i in range(len(idx_v)):
        x_lt = hs_lt[idx_v[i]:int(idx_v[i]+win_N)]
        x_sh = hs_sh[idx_v[i]:int(idx_v[i]+win_N)]
        x_act = mag[idx_v[i]:int(idx_v[i]+win_N)]
        a_resp = a[:,idx_resp[i]:int(idx_resp[i]+win_N_resp)]
        zz = z_resp[idx_resp[i]:int(idx_resp[i]+win_N_resp)]
        xx = x_resp[idx_resp[i]:int(idx_resp[i]+win_N_resp)]
        yy = y_resp[idx_resp[i]:int(idx_resp[i]+win_N_resp)]
        theta_slice = theta[idx_bo[i]:int(idx_bo[i]+win_N_bo)]
        
        # Check for previous heart rate and restrict large changes in heart rate
        if i:
            if (vitals[i-1,1] != 0) & (vitals[i-1,3] > params.activity_thresh[0]) & (vitals[i-1,3] < params.activity_thresh[1]):
                params.min_scale = int(np.floor(60/(vitals[i-1,1]*1.1)*params.fs_ds))
                params.max_scale = int(np.ceil(60/(vitals[i-1,1]*.9)*params.fs_ds))
            else:
                params.min_scale = int(np.floor(60/params.hr[1]*params.fs_ds)) # Assuming initial heart rate is non-tachycardia
                params.max_scale = int(np.ceil(60/params.hr[0]*params.fs_ds))
        
        activity = calculate_activity(x_act,params,params.activity)
        avg_theta = np.mean(theta_slice)
            
        pks_sh, pks_lt, matched, b2b, SQI = calculate_b2b(x_sh,x_lt,params)
        ac = autocorr(x_sh)
        win_hr = 0
        SQI = 0
        ac_peaks, _ = signal.find_peaks(ac, distance = 60/params.hr[1]*params.fs_ds, height = 0.2, prominence = 0.1)
        if ac_peaks.size > 0:
            win_hr = 60/(ac_peaks[0]/params.fs_ds)
            SQI = ac[ac_peaks[0]]
        else:
            SQI = 0
        # fig, (ax1, ax2) = plt.subplots(2,1)
        # ax1.plot(np.arange(len(x_sh)), x_sh)
        # ax2.plot(np.arange(len(ac)), ac)
        # ax2.scatter(ac_peaks, ac[ac_peaks])
        # ax1.set_title(f'HR = {win_hr} bpm')
        # plt.show()
        # plt.close()

        # print(matched)
        all_pks_sh = all_pks_sh + list((pks_sh + idx_v[i]))
        all_pks_lt = all_pks_lt + list((pks_lt + idx_v[i]))
        all_matched = all_matched + list((matched + idx_v[i]))
        all_matched_amp = all_matched_amp + list(x_sh[matched])
        
        if (not np.isscalar(matched)): #& (vitals[i-1,3] > params.activity_thresh[0]) & (vitals[i-1,3] < params.activity_thresh[1]) & (SQI > .8) :
            beats = np.concatenate((beats,matched+idx_v[i]))
            hs_amp = np.mean(x_sh[matched])     # HR amplitude
        else:
            hs_amp = np.nan
        
        # if 60/np.mean(b2b) > 100:
        #     plt.figure()
        #     plt.plot(b2b,'p')
        
                
    ##        wav,resp,orientation = resp_orient(a_resp,params)
    ##        wave = np.concatenate((wave,wav))
    ##        #pos = np.concatenate((pos,orientation))
    ##        #ppk = remove_spikes(wav,params.fs_resp,.05)
    ##        pks_resp,_ = signal.find_peaks(wav,distance=params.fs_resp/params.resp[1],height=.001)
    ##        rms_resp = np.std(wav)      # RR amplitude

        pks_resp,_ = signal.find_peaks(xx, distance=params.fs_resp/params.resp[1], prominence=.0002)
        # pks_resp_max, pks_resp_min = peakdetect(zz, lookahead = int(params.fs_resp/params.resp[1]/1.25))
        # pks_resp_max = np.array(pks_resp_max)[:,0]
        # pks_resp_min = np.array(pks_resp_min)[:,0]
        rms_resp = np.std(zz)      # RR amplitude

        # all_pks_resp = all_pks_resp + list((pks_resp_max + idx_resp[i])/params.fs_resp) + list((pks_resp_min + idx_resp[i])/params.fs_resp)
        all_pks_resp = all_pks_resp + list((pks_resp + idx_resp[i]))


        vitals[i,3] = activity # Even if len(b2b)=0, activity should be calculated, not thrown out.
        vitals[i,8] = avg_theta
        
        if (len(b2b)>1):
            vitals[i,1] = 60/np.mean(b2b) # Heart Rate
            vitals[i,2] = SQI # Signal Quality Index
            vitals[i,4] = 60*params.fs_resp/np.mean(np.diff(pks_resp))#60/np.mean(resp) # Respiration Rate
            # vitals[i,4] = 60*params.fs_resp/((np.mean(np.diff(pks_resp_max)) + np.mean(np.diff(pks_resp_min)))/2)
            if vitals[i,4] < params.resp[0]*60 or vitals[i,4] > params.resp[1]*60:
                vitals[i,4] = -10000
            vitals[i,5] = np.sqrt(1/(len(b2b)-1)*np.sum(np.diff(b2b)**2))
            vitals[i,6] = hs_amp
            vitals[i,7] = rms_resp
        
        if i%10 == 0:
            print(f"PROCESSED VITALS WINDOW {i+1} of {len(idx_v)}")


    all_matched = np.array(all_matched)
    print(all_matched.shape)
    all_matched, unique_index = np.unique(all_matched, return_index=True)

    all_matched_amp = np.array(all_matched_amp)
    all_matched_amp = all_matched_amp[unique_index]
    print(all_matched.shape, all_matched_amp.shape)

    delta_S1 = np.diff(all_matched)
    print(len(all_matched), len(delta_S1))

    all_pks_sh = np.array(all_pks_sh)
    all_pks_sh, unique_index = np.unique(all_pks_sh, return_index=True)

    all_pks_lt = np.array(all_pks_lt)
    all_pks_lt, unique_index = np.unique(all_pks_lt, return_index=True)


    all_pks_resp = np.array(all_pks_resp)
    all_pks_resp, unique_index = np.unique(all_pks_resp, return_index=True)

    all_matched = all_matched/params.fs_ds
    all_pks_sh = all_pks_sh/params.fs_ds
    all_pks_lt = all_pks_lt/params.fs_ds
    all_pks_resp = all_pks_resp/params.fs_resp

    ################RR by heartbeat modulation effect############
    all_matched_idx = all_matched * params.fs
    all_matched_idx = all_matched_idx.astype(int)

    for i in range(len(all_matched_idx)):
        all_matched_idx[i] = np.argmax (accel_z_clean[all_matched_idx[i] - 8 : all_matched_idx[i] + 8]) + (all_matched_idx[i] - 8 )

    all_matched_idx, unique_index = np.unique(all_matched_idx, return_index=True)

    cs = CubicSpline(all_matched_idx, accel_z_clean[all_matched_idx])
    all_matched_idx_interp = np.linspace(all_matched_idx[0], all_matched_idx[-1], (all_matched_idx[-1] - all_matched_idx[0]+1))
    contour_interp = cs(all_matched_idx_interp)

    sos_RR = signal.butter(4,np.array([1.25,3.0])/(params.fs/2),btype='bandpass',output='sos')
    sos_RR_2 = signal.butter(4, 0.1/(params.fs/2),btype='high',output='sos')
    contour_interp_filt = signal.sosfiltfilt(sos_RR,contour_interp)
    resp_peaks, _ = signal.find_peaks(contour_interp_filt, distance = params.fs/params.resp[1], prominence = 0.0005)

    win_N_resp = params.vital_w*params.fs
    idx_resp = np.arange(0,int(len(all_matched_idx_interp)-win_N_resp),int(np.floor(win_N_resp*params.vital_ovlp)))

    all_pks_resp_heart = []
    for i in range(len(idx_resp)):
        xx = contour_interp_filt[idx_resp[i]:int(idx_resp[i]+win_N_resp)]
        pks_resp,_ = signal.find_peaks(xx,distance=params.fs/params.resp[1],prominence=.0005)
        # print(pks_resp)
        all_pks_resp_heart = all_pks_resp_heart + list((pks_resp + idx_resp[i]))
        vitals[i,9] = 60*params.fs/np.mean(np.diff(pks_resp))#60/np.mean(resp) # Respiration Rate
        if vitals[i,9] < params.resp[0]*60 or vitals[i,9] > params.resp[1]*60:
            vitals[i,9] = -10000
    all_pks_resp_heart = np.array(all_pks_resp_heart)
    all_pks_resp_heart, unique_index = np.unique(all_pks_resp_heart, return_index=True)
    all_pks_resp_heart = all_pks_resp_heart/params.fs

    return vitals, accel_z_clean, hs, hs_sh, hs_lt, all_pks_sh, all_pks_lt, all_matched,  x_resp, all_pks_resp, all_pks_resp_heart

# directory = sys.argv[1]
# print(directory)
directory = '/Users/wouyang/OneDrive/MA/Surgeries/19th_surgery_10152022/Witness_Defeat/Male/021423_MA188_in_pain_MA189/MA188_23-02-14_12_08_10/'
N_file = len(glob(directory + 'aligned_parsed_data_*.csv'))
print(N_file)


for kk in range(N_file):
    path = directory + f'aligned_parsed_data_{kk+1}.csv'
    data = pd.read_csv(path).to_numpy()

    params = paramstruct
    params.fs = 800 #np.floor(1000 / np.median(np.diff(data['time(ms)'])))
    scale = 16384

    temp_time = data[~np.isnan(data[:, 9]), 1]
    temp = data[~np.isnan(data[:, 9]), 9]
    temp_export = np.zeros([temp.shape[0], 2])
    temp_export[:, 0] = temp_time
    temp_export[:, 1] = temp
    # print(temp_export)
    np.save(path[:-4]+'_temp.npy', temp_export)

    imu1_time = data[~np.isnan(data[:, 3]), 1]
    imu1_x = data[~np.isnan(data[:, 3]), 3]
    imu1_y = data[~np.isnan(data[:, 3]), 4]
    imu1_z = data[~np.isnan(data[:, 3]), 5]

    ###because of disconnection, imu data are separated into chunks, need to find the chunks and process separately#########
    imu1_time_diff = np.diff(imu1_time)
    chunk_end = np.argwhere(imu1_time_diff > 1000/params.fs*1.1)
    chunk_start = [0]
    chunk_start = np.append(chunk_start, chunk_end + 1)
    chunk_end = np.append(chunk_end, len(imu1_time_diff))
    # print(chunk_start, chunk_end)

    for kkk in range(len(chunk_start)):
        print(f"Processing data {kk+1}, chunk {kkk+1}\n")
        if chunk_end[kkk] - chunk_start[kkk] > params.fs * 60:  # drop chunks below 1 min
            # convert from ms to s
            chunk_time = imu1_time[chunk_start[kkk]:chunk_end[kkk]] / 1000
            start_time = chunk_time[0]
            accel_x = imu1_x[chunk_start[kkk]:chunk_end[kkk]]
            accel_y = imu1_y[chunk_start[kkk]:chunk_end[kkk]]
            accel_z = imu1_z[chunk_start[kkk]:chunk_end[kkk]]

            # accel_x, accel_y, accel_z = clean_data(accel_x, accel_y, accel_z)
            accel_x = accel_x/scale
            accel_y = accel_y/scale
            accel_z = accel_z/scale
            chunk_time = start_time + np.arange(len(accel_x))/params.fs

            # Calculate Parameters
            params.total_time = len(accel_z) / params.fs
            # print(params.total_time)
            params.start_time = start_time  # data.iloc[0]['time(ms)']
            # data.iloc[-1]['time(ms)']
            params.end_time = start_time+params.total_time

            # Pre-process, filter data and calculate transforms

            vitals, accel_z_clean, hs, hs_sh, hs_lt, all_pks_sh, all_pks_lt, all_matched, \
            x_resp, all_pks_resp, all_pks_resp_heart = calculate_vitals(params, accel_x, accel_y, accel_z)

            time = vitals[:, 0]
            HR = vitals[:, 1]
            SQI = vitals[:, 2]
            PA = vitals[:, 3]
            RR = vitals[:, 4]
            BO = vitals[:, 8]
            RR_heart = vitals[:,9]

            idx_good = (SQI >= .0)

            # print(vitals)

            fig, (axx1, axx2, axx3, axx4, axx5, axx6) = plt.subplots(
                6, 1, sharex=True)

            axx1.plot(chunk_time, accel_z, c='b', linewidth=0.5)
            axx1.plot(chunk_time, accel_x, c='k', linewidth=0.5)
            axx1.plot(chunk_time, accel_y, c='r', linewidth=0.5)
            ##axx1.plot(np.arange(len(z_resp))/params.fs_resp/60, z_resp, linewidth = 0.5)

            axx2.plot(chunk_time, accel_z_clean, c='b', linewidth = 0.5)

            # axx2.plot(np.arange(len(theta))/params.fs_bo/60,theta)
            axx3.scatter(time[idx_good], HR[idx_good], c='r', s=3, marker='o')
            axx4.scatter(time[idx_good], RR[idx_good], c='b', s=3, marker='o')
            axx5.plot(time, PA+0.000001)
            axx6.scatter(time[idx_good], SQI[idx_good], c='k', s=3, marker='o')

            axx1.set_xlim([chunk_time[0], chunk_time[-1]])
            axx1.set_ylim([-2, 2])
            axx1.set_ylabel('Accel_z\n(g)')

            axx2.set_xlim([chunk_time[0], chunk_time[-1]])
            axx2.set_ylim([-2, 2])
            axx2.set_ylabel('Accel_z\n(g)')

            axx3.set_ylim([400, 800])
            axx3.set_yticks([400, 600, 800])
            axx3.set_ylabel('HR (BPM)')
            #axx3.legend(loc='upper right',ncol=2, columnspacing = 0.5, handlelength = 0.2, fontsize = 'x-small')

            axx4.set_ylim([60, 140])
            axx4.set_yticks([60, 100, 140])
            axx4.set_ylabel('RR (BPM)')

            axx5.set_ylim([0.0, 0.1])
            axx5.set_yticks([0.0, 0.05, 0.1])
            axx5.set_ylabel('Physical\nactivity (a.u.)')

            axx6.set_ylim([0, 1.0])
            axx6.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            axx6.set_ylabel("SQI\n")
            axx6.set_xlabel('Time (s)')

            ticks = np.arange(chunk_time[0], chunk_time[-1], 60)
            tick_labels = [datetime.fromtimestamp(tick+5*3600).strftime("%H:%M:%S") for tick in ticks]
            axx6.set_xlim([chunk_time[0], chunk_time[-1]])
            axx6.set_xticks(ticks)
            axx6.set_xticklabels(tick_labels)
            axx6.set_xlabel('Time (HH:MM)')

            fig.tight_layout()
            fig.align_ylabels()

            # plt.show()

            fig.savefig(path[:-4] + f'savgol_{start_time}.png', dpi=300)
            plt.close(fig)

            np.save(path[:-4] + f'savgol_{start_time}_vitals.npy', vitals)

            accel_export = np.hstack((chunk_time.reshape((len(chunk_time), 1)), accel_x.reshape((len(chunk_time), 1)), accel_y.reshape((len(chunk_time), 1)), accel_z.reshape((len(chunk_time), 1)), accel_z_clean.reshape((len(chunk_time), 1))))
            np.save(path[:-4] + f'savgol_{start_time}_accel.npy', accel_export)

           