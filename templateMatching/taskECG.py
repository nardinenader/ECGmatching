import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

def windowMethod(StopBand, TransitionBand, frequencySampling):
    if 44 < StopBand <= 53:  # hamming
        delta_f = TransitionBand / frequencySampling
        N = 3.3 / delta_f
        if N % 2 == 0:
            N += 1
        else:
            N = round(N)
            if N % 2 == 0:
                N += 1
        n = np.arange(-int((N - 1) / 2), int((N - 1) / 2) + 1)
        print("the N:", N)
        print("the range:", n)
        window_function = 0.54 + 0.46 * np.cos((2 * np.pi * n) / (N))

        print("the window function:", window_function)
        print("hamming")

        return window_function, n

def bandpass(StopBand, TransitionBand, frequencySampling,fcnew, fcnew2):
    window_function, n = windowMethod(StopBand, TransitionBand, frequencySampling)
    HD = np.zeros_like(n, dtype=float)
    fc_new1 = (fcnew - (TransitionBand / 2)) / frequencySampling
    fc_new2 = (fcnew2 + (TransitionBand / 2)) / frequencySampling
    angular_fc1 = 2 * np.pi * fc_new1
    angular_fc2 = 2 * np.pi * fc_new2
    xlen = len(n) // 2
    HD[0 + xlen] = 2 * (fc_new2 - fc_new1)

    for i, n_value in enumerate(n):
        if n_value != 0:
            HD[i] = (2 * fc_new2 * (np.sin(n_value * angular_fc2) / (n_value * angular_fc2))) - (
                    2 * fc_new1 * np.sin(n_value * angular_fc1) / (n_value * angular_fc1))

    final_result = HD * window_function

    return final_result

def convolve(filter_signal, input_signal):
    len_filter = len(filter_signal)
    len_input = len(input_signal)
    len_convolved = len_filter + len_input - 1

    convolved_signal = np.zeros(len_convolved)

    # Perform convolution
    for i in range(len_filter):
        for j in range(len_input):
            convolved_signal[i + j] += filter_signal[i] * input_signal[j]

    return convolved_signal


def safe_resample(signal, original_fs, new_fs):

    duration = len(signal) / original_fs  # Duration of the signal in seconds
    num_samples_new = int(duration * new_fs)  # Number of samples in the resampled signal

    if num_samples_new > 1:
        # Create a new signal with the desired number of samples
        resampled_signal = []
        for i in range(num_samples_new):
            # Find the corresponding index in the original signal
            original_index = i * (len(signal) / num_samples_new)

            # If the index is an integer, use the value from the original signal
            if original_index.is_integer():
                resampled_signal.append(signal[int(original_index)])
            else:
                # Interpolate between the two surrounding points in the original signal
                lower_index = int(original_index)
                upper_index = min(lower_index + 1, len(signal) - 1)  # Ensure index is within bounds
                weight = original_index - lower_index  # Weight for interpolation

                # Linear interpolation
                interpolated_value = (1 - weight) * signal[lower_index] + weight * signal[upper_index]
                resampled_signal.append(interpolated_value)

        return resampled_signal
    else:
        print("newFs is not valid")
        return signal


def remove_f_dc(resmapled_signal):
    dc_signal  = resmapled_signal
    all_dc_removed_data = []
    freq_signal = np.fft.fft(dc_signal)
    freq_signal[0] = 0
    # freq_signal = freq_signal[1:]
    dc_removed_data = np.fft.ifft(freq_signal)
    np.set_printoptions(precision=4, suppress=True)
    d = np.real(dc_removed_data)
    return dc_removed_data


def normalize(dc_removed_result):
    min_val = np.min(dc_removed_result)
    max_val = np.max(dc_removed_result)

    normalized_data = 2 * ((dc_removed_result - min_val) / (max_val - min_val)) - 1
    return normalized_data

def shift_left(array, positions):
    positions = positions % len(array)  # Ensures positions is within the valid range
    return np.concatenate((array[positions:], array[:positions]))

def Correlation(normalized_signal, Second_Signal):
    signals = normalized_signal
    N = len(signals)
    Cor = [0] * N
    for i in range(N):
        currentShift = shift_left(Second_Signal, i)
        for j in range(N):
            Cor[i] = Cor[i] + (signals[j] * currentShift[j])

        Cor[i] = Cor[i] * 1 / N

    return Cor


def compute_dct(correlated_signal):
    N = len(correlated_signal)
    dct_result = np.zeros(N)

    for k in range(N):
        sum_val = 0.0
        for n in range(N):
            sum_val += correlated_signal[n] * np.cos((np.pi / (4 * N)) * (2 * n - 1) * (2 * k - 1))

        dct_result[k] = np.sqrt(2 / N) * sum_val

    return dct_result

def getAverage(Subjects):
    N = len(Subjects[0])
    avgClass = [0] * N
    for i in range(N):
        for j in range(6):
            avgClass[i] = avgClass[i] + Subjects[j][i]

        avgClass[i] = avgClass[i] / len(Subjects)

    return avgClass

def load_text_file(file_path):

    try:
        # Assuming the file contains one floating point number per line
        with open(file_path, 'r') as file:
            data = [float(line.strip()) for line in file]
            print("LOADED")
        return data
    except Exception as e:
        print(f"An error occurred while loading the file {file_path}: {e}")
        return []


#run_all(a,b,test,stopband,transitioband,fs,fsnew,f1,f2)
def run_all(a,b,test):
    #

    # Load the ECG data for each subject from their respective files
    ECG_SUBJECTSA = a
    ECG_SUBJECTSB = b
    ECG_TEST = test

    StopBand = 50
    TransitionBand = 500
    frequencySampling = 1000
    new_FS = 100
    fc1 = 150
    fc2 = 250

    # since window and bandpass is the same it will out of the loop
    bandpass_result = bandpass(StopBand, TransitionBand, frequencySampling, fc1, fc2)

    ECG_SUBJECTSA_afer_preprocessing = []
    ECG_SUBJECTSB_afer_preprocessing = []
    ECG_TEST_afer_preprocessing = []

    for signal in ECG_SUBJECTSA:
        convolve_result = convolve(bandpass_result, signal)
        resampled_signal_result = safe_resample(convolve_result, frequencySampling, new_FS)
        remove_dc_result = remove_f_dc(resampled_signal_result)
        normalized_result = normalize(remove_dc_result)
        correlation_result = Correlation(normalized_result, normalized_result)
        DCT_result = compute_dct(convolve_result)
        ECG_SUBJECTSA_afer_preprocessing.append(DCT_result)

    # For SUBJECT B

    for signal in ECG_SUBJECTSB:
        convolve_result = convolve(bandpass_result, signal)
        resampled_signal_result = safe_resample(convolve_result, frequencySampling, new_FS)
        remove_dc_result = remove_f_dc(resampled_signal_result)
        normalized_result = normalize(remove_dc_result)
        correlation_result = Correlation(normalized_result, normalized_result)
        DCT_result = compute_dct(convolve_result)
        ECG_SUBJECTSB_afer_preprocessing.append(DCT_result)

    # FOR TEST

    for signal in ECG_TEST:
        convolve_result = convolve(bandpass_result, signal)
        resampled_signal_result = safe_resample(convolve_result, frequencySampling, new_FS)
        remove_dc_result = remove_f_dc(resampled_signal_result)
        normalized_result = normalize(remove_dc_result)
        correlation_result = Correlation(normalized_result, normalized_result)
        DCT_result = compute_dct(convolve_result)
        ECG_TEST_afer_preprocessing.append(DCT_result)

    # Get Average of Each subject
    ECG_SUBJECTSA_average = getAverage(ECG_SUBJECTSA_afer_preprocessing)
    ECG_SUBJECTSB_average = getAverage(ECG_SUBJECTSB_afer_preprocessing)

    # Made it [0] cuz the way we load files is 2d even if its only 1 file change later >>
    Correlation_SUBJECTA_with_TEST = Correlation(ECG_SUBJECTSA_average, ECG_TEST_afer_preprocessing[0])
    Correlation_SUBJECTB_with_TEST = Correlation(ECG_SUBJECTSB_average, ECG_TEST_afer_preprocessing[0])

    if Correlation_SUBJECTA_with_TEST[0] > Correlation_SUBJECTB_with_TEST[0]:
        print("Test is Subject A")
    else:
        print("Test is Subject B")

