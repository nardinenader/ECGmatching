import cmath
import tkinter as tk
from tkinter import filedialog, font
import numpy
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
from tabulate import tabulate
from decimal import Decimal

import taskECG

frequencies = []
indeces = []
chosen_letter = 0
chosen_number = 0
rows_to_skip = 3
signals = []
normalized_data = None
shifted_data = None
assigned_midpoints = []
quantization_error = []
encoded_signals = []
indexes = []
sampling_frequency = 0

# /////////////////////////////////////////file reading
def Compare_Signals(file_name,Your_indices,Your_samples):
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one")
            return
    print("Test case passed successfully")

def read_samples_from_file(file_path):
    samples = np.loadtxt(file_path, skiprows=rows_to_skip)
    return samples

def read_files():

    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])
    if len(file_paths) >= 1:
        signals.clear()  # Clear the signals list

        # Create subplots outside the loop
        fig, axes = plt.subplots(1, len(file_paths), figsize=(12, 4))

        for i, file_path in enumerate(file_paths):
            with open(file_path, 'r') as file:
                signal = np.loadtxt(file, skiprows=rows_to_skip)
                signals.append(signal[:, 1])

        #         # Plot each signal and set titles
        #         axes[i].plot(signal[:, 1])
        #         axes[i].set_title(f'Signal {i + 1}')
        #
        # # Adjust spacing between subplots
        # plt.tight_layout()
        #
        # plt.show()

def read_files2():
        global signals
        input_text = con4.get()
        sampling_frequency = int(input_text)
        file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])
        if len(file_paths) >= 1:
            signals.clear()  # Clear the signals list
            amplitude2 =[]
            phase = []
            for file_path in file_paths:
                with open(file_path, 'r') as file:
                    data = np.loadtxt(file, skiprows= rows_to_skip)
                    time = data[:, 0]
                    amplitude = data[:, 1]
                    # amplitude2 = float(time)
                    # phase= float(amplitude)
                    amplitude2.append(time)
                    phase.append(amplitude)

                    signal = amplitude * np.cos(2 * np.pi * time * sampling_frequency)
                    signals.append(signal)
            return amplitude2,phase
def read_files3():

    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])
    signals.clear()



    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r') as file:
            signal = np.loadtxt(file, )
            signals.append(signal)



# //////////////task 1
def eq():
    soc_entry_value = soc_entry.get()
    amp = float(amp_entry.get())
    analog_frequency = float(analog_entry.get())
    phase_shift = float(shift_entry.get())
    sampling_frequency = float(sampling_entry.get())

    t_continuous = np.linspace(0, 1, 1000)  # Time axis for continuous signal
    t_discrete = np.linspace(0, 99, 100)  # Discrete time axis

    if soc_entry_value == 'sin':
        y_continuous = amp * np.sin(2 * np.pi * analog_frequency * t_continuous + phase_shift)
        y_discrete = amp * np.sin(2 * np.pi * analog_frequency / sampling_frequency * t_discrete + phase_shift)
    elif soc_entry_value == 'cos':
        y_continuous = amp * np.cos(2 * np.pi * analog_frequency * t_continuous + phase_shift)
        y_discrete = amp * np.cos(2 * np.pi * analog_frequency / sampling_frequency * t_discrete + phase_shift)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t_continuous, y_continuous)
    plt.title('Continuous Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.subplot(1, 2, 2)
    plt.stem(t_discrete, y_discrete)
    plt.title('Discrete Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

def plot_signal(signal):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(signal[:, 1])
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Continuous Signal')

        plt.subplot(1, 2, 2)
        plt.stem(signal[:, 1])
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.title('Discrete Signal')
        plt.tight_layout()
        plt.show()

def button_click():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
     samples = read_samples_from_file(file_path)
     plot_signal(samples)

# ///////////////////task 2
def plotting(var, _title):
            plt.figure(figsize=(10, 4))
            plt.plot(var)
            plt.title(_title)
            plt.show()


def add():
    if len(signals) >= 2:
        result = signals[0] + signals[1]
        t = "Addition of signals"
        plotting(result, t)


def subtract():
    if len(signals) >= 2:
        result = signals[1] - signals[0]
        t = "Subtraction of signals"
        plotting(result, t)


def normalization():
    global normalized_data
    if len(signals) >= 1:
      normalize_range = float(normval.get())
      min_val = np.min(signals[0])
      max_val = np.max(signals[0])

    if normalize_range == 1:
        normalized_data = 2 * ((signals[0] - min_val) / (max_val - min_val)) - 1

    elif normalize_range == 2:
        normalized_data = (signals[0] - min_val) / (max_val - min_val)
    t = "Normalized Signal"
    plotting(normalized_data, t)


def shift():
    global shifted_data
    if len(indeces) >= 1:
        constant = int(shift_value.get())
        if constant > 0:
            shifted_data = indeces[0] - constant  # Shift up
        elif constant < 0:
            shifted_data = indeces[0] + constant  # Shift down

    t = "Shifted Signal"
    plotting(shifted_data, t)
def multiplication():
        file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])
        if len(file_paths) == 1:
            signals = []
            for file_path in file_paths:
                with open(file_path, 'r') as file:
                    lines = file.readlines()[3:]
                    values = [float(line.split()[1]) for line in lines]
                    multiplier = int(con3.get())
                    multiplied_values = [value * multiplier for value in values]
                    signals.append(multiplied_values)

            if len(signals) == 1:
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.plot(signals[0])
                plt.title('Signal')
                plt.tight_layout()
                plt.show()


def squaring():
 if len(signals) >= 1:
     squared_signal = signals[0] ** 2
     t = "squared signal"
     plotting(squared_signal, t)


def acc():
    accumulation = np.cumsum(signals[0])
    t = "Accumulation of the signal"
    plotting(accumulation, t)

#/////////////////////////////task 3
def quantization(file_paths,entry_letter_value, entry_number_value):
    global indexes
    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", ".txt")])
    if len(file_paths) == 1:
        _signals = []
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                lines = file.readlines()[3:]
                values = [float(line.split()[1]) for line in lines]
                _signals.extend(values)  # Use extend to add values to the _signals list

        min_signal = np.min(_signals)
        max_signal = np.max(_signals)
        choice = entry_letter_value


        if choice.lower() == 'b':
            value = int(entry_number_value)
            level = 2 ** value

        elif choice.lower() == 'l':
            valuel = int(entry_number_value)
            value = int(np.log2(valuel))

            level = valuel  # Assign level using user input value

        delta = (max_signal - min_signal) / level
        ranges = [[min_signal + i * delta, min_signal + (i + 1) * delta] for i in range(level)]
        ranges[-1][-1] = max_signal  # Update the last element of the last range to be equal to max_value

        # midpoint:
        midpoints = [(range[0] + range[1]) / 2 for range in ranges]




        for signal in _signals:
            assigned_midpoint = None
            for i, _range in enumerate(ranges):
                if _range[0] <= signal <= _range[1]:
                    assigned_midpoint = midpoints[i]
                    encoded_signal = format(i, '0'+str(value)+'b')  # Encode the signal using the number of bits
                    indexes.append(i+1)
                    encoded_signals.append(encoded_signal)
                    break  # Exit the loop if a range is found for the signal
            assigned_midpoints.append(float("{:.3f}".format(assigned_midpoint)))
            quantization_error.append(float("{:.3f}".format(assigned_midpoint - signal)))


def display2():
    #global encoded_signals, quantization_error, assigned_midpoints
    global test_area

    result_text = "Quantization Error:\n" + ", ".join(map(str, quantization_error)) + "\n\n"
    result_text += "Encoded Signal:\n" + ", ".join(map(str, encoded_signals)) + "\n\n"
    result_text += "Ranges Interval:\n" + ", ".join(map(str, indexes)) + "\n\n"
    result_text += "Assigned Midpoint:\n" + ", ".join(map(str, assigned_midpoints))

    # Clear the text area and insert the new text
    # text_area.delete(1.0, tk.END)
    # text_area.insert(tk.END, result_text)

#
def display():

    global test_area
    result_text = "Encoded Signal:\n" + ", ".join(map(str, encoded_signals)) + "\n\n"
    result_text += "Assigned Midpoint:\n" + ", ".join(map(str, assigned_midpoints))

    # Clear the text area and insert the new text
    # text_area.delete(1.0, tk.END)
    # text_area.insert(tk.END, result_text)

def submit():
    letter_value = entry_letter.get()
    number_value = entry_number.get()
    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", ".txt")])
    quantization(file_paths, letter_value, number_value)
    print(letter_value)


#////////////////////////////////////task 4
signals = []

def Fourier():
    global amplitude
    global frequencies
    global phase
    global sampling_frequency
    input_text = con4.get()
    sampling_frequency = int(input_text)
    t = np.arange(0, 1, 1 / sampling_frequency)

    signals_combined = np.concatenate(signals)

    N = len(signals_combined)
    frequencies = np.fft.fftfreq(N, d=1/sampling_frequency)

    amplitude = np.zeros(N)
    phase = np.zeros(N)

    for k in range(N):
        real_part = np.sum(signals_combined * np.cos(2 * np.pi * frequencies[k] * t))
        imag_part = -np.sum(signals_combined * np.sin(2 * np.pi * frequencies[k] * t))

        amplitude[k] = np.sqrt(real_part * 2 + imag_part * 2)
        phase[k] = np.arctan2(imag_part, real_part)


    for freq, amp, ph in zip(frequencies, amplitude, phase):
        print(f"{amp}f {ph}f")

    outputfile = "C:\\Users\\kenzy\\Desktop\\semester1materials\\digital signal processing\\readingofpolar.txt"
    with open(outputfile, "w") as file:
        file.write("0\n1\n8\n")
        for freq, amp, ph in zip(frequencies, amplitude, phase):
            # Format the output string
            output_str = f"{amp} {ph}\n"
            # Write the formatted string to the file
            file.write(output_str)

    plt.figure(figsize=(12, 6))

    # Frequency versus amplitude
    plt.subplot(121)
    plt.stem(frequencies, amplitude)
    plt.title("Frequency vs. Amplitude")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")

    # Frequency versus phase
    plt.subplot(122)
    plt.stem(frequencies, phase)
    plt.title("Frequency vs. Phase")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")

    plt.tight_layout()
    plt.show()

def modify_signal():
    global frequencies
    global amplitude
    global phase_shift
    global sampling_frequency
    # Read user-input amplitude and phase shift
    print("eshtaeshta")
    amplitude = float(entry_amplitude.get())
    phase = float(entry_phase.get())

    # Modify the existing signals with the new amplitude and phase shift
    for i, signal in enumerate(signals):
        # Modify the signal
        modified_signal = signal * amplitude * np.cos(
            2 * np.pi * frequencies[i] * np.arange(0, 1, 1 / float(sampling_frequency)) + phase)
        signals[i] = modified_signal


    Fourier()
    Invers_Fourier()

def Invers_Fourier():
    amp,phis = read_files2()
    appended_amp = []
    appended_phis = []
    for i in amp:
        for j in i:
           appended_amp.append(j)
    for i in phis:
        for j in i:
            appended_phis.append(j)
    harmonicX = []
    for i in range(len(appended_amp)):
        com = appended_amp[i]*np.exp(1j*appended_phis[i])
        harmonicX.append(com)

    N = len(appended_amp)
    inverse_transform = []
    x = []
    for n in range(N):
        x.append(n)
        result = 0
        for k in range(N):
            term = np.round(harmonicX[k] * cmath.exp(2j * cmath.pi * n * k / len(appended_amp)), 5)
            result += term
        inverse_transform.append(result.real/len(appended_amp))

    print(inverse_transform)

    plt.figure(figsize=(12, 6))

    inverse_transform_real = np.real(inverse_transform)

    plt.plot(x, inverse_transform)
    plt.title("Inverse Fourier Transform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

# /////////////////////////////////////////////////////////task 6

def smoothing():
    avgSize = float(entry_avg.get())
    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])
    mov_averages = []

    if len(file_paths) >= 1:
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                smooth_data = np.loadtxt(file, skiprows=rows_to_skip)[:, 1:]

            for n in range(len(smooth_data) - int(avgSize) + 1):
                # Calculate the average of the current window
                window = round(np.sum(smooth_data[n:n + int(avgSize)]) / int(avgSize), 6)

                # Store the average of the current window in the moving average list
                mov_averages.append(window)


        t = "Smoothing Signal"
        plotting(mov_averages, t)



def shifting(k, foldedSignal):
    k=0
    shiftedData=0

    if k > 0:
        shiftedData = foldedSignal - k
    elif k < 0:
        shiftedData = foldedSignal + k

    return shiftedData


def DerivativeSignal():
    InputSignal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                   19.0, 20.0, 21.0, 22.0,
                   23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0,
                   40.0, 41.0, 42.0,
                   43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0,
                   60.0, 61.0, 62.0,
                   63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                   80.0, 81.0, 82.0,
                   83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0,
                   100.0]

    expectedOutput_first = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1]
    expectedOutput_second = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0]

    FirstDrev = [InputSignal[n] - InputSignal[n - 1] for n in range(1, len(InputSignal))]

    SecondDrev = [InputSignal[n + 1] - 2 * InputSignal[n] + InputSignal[n - 1] for n in range(1, len(InputSignal)-1)]

    if ((len(FirstDrev) != len(expectedOutput_first)) or (len(SecondDrev) != len(expectedOutput_second))):
        print("mismatch in length")
        return
    first = second = True
    for i in range(len(expectedOutput_first)):
        if abs(FirstDrev[i] - expectedOutput_first[i]) < 0.01:
            continue
        else:
            first = False
            print("1st derivative wrong")
            return
    for i in range(len(expectedOutput_second)):
        if abs(SecondDrev[i] - expectedOutput_second[i]) < 0.01:
            continue
        else:
            second = False
            print("2nd derivative wrong")
            return
    if (first and second):
        print("Derivative Test case passed successfully")
    else:
        print("Derivative Test case failed")
    return


def shiftFold():
    k = float(entry_k.get())
    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])

    if len(file_paths) >= 1:
        for i, file_path in enumerate(file_paths):
            with open(file_path, 'r') as file:
                f = np.loadtxt(file, skiprows=rows_to_skip)[:, 0:]

    shifting(k, f)
    print(f)
    t = "shifted Folded data"
    plotting(f, t)

def folding():
    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])

    if len(file_paths) >= 1:
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                preFolded = np.loadtxt(file, skiprows=rows_to_skip)[:, 1:]

    folded_signal = np.flip(preFolded)
    print(np.real(folded_signal))
    t = "folded signal"
    plotting(folded_signal, t)
    return folded_signal


def remove_f_dc():


        file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])
        all_dc_removed_data = []

        if len(file_paths) >= 1:
            for i, file_path in enumerate(file_paths):
                with open(file_path, 'r') as file:
                    dc_signal = np.loadtxt(file, skiprows=rows_to_skip)[:, 1:]


                    # Perform FFT on the signal
                    freq_signal = np.fft.fft(dc_signal)
                    freq_signal[0] = 0
                    # freq_signal = freq_signal[1:]
                    dc_removed_data = ifft(freq_signal)
                    np.set_printoptions(precision=4, suppress=True)
                    print (dc_removed_data)
                    d = np.real(dc_removed_data)
                    print(d)
                    # print(dc_signal)


                    t = "Signal without DC component"
                    plotting(d, t)

                    all_dc_removed_data.append(dc_removed_data)




#/////////////////////////////////////////////////////////////////////////////////////task 7
global avgClass1
global avgClass2
def shift_left(array, positions):
    positions = positions % len(array)  # Ensures positions is within the valid range
    return np.concatenate((array[positions:], array[:positions]))

def Correlation():
    X1 = signals[0]
    X2 = signals[1]
    N = len(signals[0])
    Cor = [0]*N
    for i in range (N):
        currentShift = shift_left(X2,i)
        for j in range (N):
            Cor[i] = Cor[i] + (X1[j]*currentShift[j])

        Cor[i] = Cor[i] * 1/N
    # Normalization start from here
    x1 = np.sum(X1 ** 2)
    x2 = np.sum(X2**2)

    Dem = np.sqrt(x1 * x2) *1/ N
    Norm = [0]*N
    for i in range(N):
        Norm[i] = Cor[i] / Dem

    print(Norm)

def Time_analysis():
#first step--> correlation
    X1 = signals[0]
    X2 = signals[1]
    N = len(signals[0])
    Cor = [0] * N
    for i in range(N):
        currentShift = shift_left(X2, i)
        for j in range(N):
            Cor[i] = Cor[i] + (X1[j] * currentShift[j])

        Cor[i] = Cor[i] * 1 / N
#second step --->el max abs value
    max_cor = np.max(Cor)
    print("the max abs value:",max_cor)
#third step ---> lag
    lag = np.argmax(np.abs(Cor))
    print("lag:",lag)
#last step ---> time delay
    fs = con5.get()
    fs2 = int(fs)
    time_delay = lag / fs2
    print ("time delay in seconds:",time_delay)


def template_matching():
    global avgClass1
    N = 251
    avgClass1 = [0]*N
    test = signals
    for i in range(251):
        for j in range(5):
            avgClass1[i] = avgClass1[i] + signals[j][i]

        avgClass1[i] = avgClass1[i]/5

def template_matching2():
    global avgClass2
    N = 251
    avgClass2 = [0] * N
    test = signals
    for i in range(251):
        for j in range(5):
            avgClass2[i] = avgClass2[i] + signals[j][i]

        avgClass2[i] = avgClass2[i] / 5

def CorWithTest():
    global avgClass1
    global avgClass2

    X2 = signals[0]
    N = len(signals[0])
    CorOfC1T1 = [0] * N
    for i in range(N):
        currentShift = shift_left(X2, i)
        for j in range(N):
            CorOfC1T1[i] = CorOfC1T1[i] + (avgClass1[j] * currentShift[j])

        CorOfC1T1[i] = CorOfC1T1[i] * 1 / N

    CorOfC2T1 = [0] * N
    for i in range(N):
        currentShift = shift_left(X2, i)
        for j in range(N):
            CorOfC2T1[i] = CorOfC2T1[i] + (avgClass2[j] * currentShift[j])

        CorOfC2T1[i] = CorOfC2T1[i] * 1 / N

    if CorOfC1T1[0] > CorOfC2T1[0]:
        print("Test is down signal")
    else:
        print("Test is up signal")

root = tk.Tk()
pad_x = 5
label_font = font.Font(size=20)
row_counter = 0

def convolve():
    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])

    if len(file_paths) == 1:
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                _signalOne = np.loadtxt(file)

    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])
    if len(file_paths) >= 1:
        signals.clear()


        fig, axes = plt.subplots(1, len(file_paths), figsize=(12, 4))

        for i, file_path in enumerate(file_paths):
            with open(file_path, 'r') as file:
                signal = np.loadtxt(file)
                signals.append(signal[:, 1])

    len_signal1 = len(_signalOne)
    len_signal2 = len(_signalOne)
    len_convolved = len_signal1 + len_signal2 - 1

    convolved_signal = np.zeros(len_convolved)

    # Perform convolution
    for i in range(len_signal1):
        for j in range(len_signal2):
            convolved_signal[i + j] += _signalOne[i][0] * signal[j][0]

    print(convolved_signal)
    t = "convolved signal"
    plotting(convolved_signal, t)


def convolveA():

    primary_file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    with open(primary_file_path, 'r') as file:
        primary_signal = np.loadtxt(file)[:,1]


    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])


    convolved_signalsA = []


    for file_path in file_paths:
        with open(file_path, 'r') as file:
            other_signal = np.loadtxt(file)

        # Calculate the length of the convolved signal
        len_primary = len(primary_signal)
        len_other = len(other_signal)
        len_convolved = len_primary + len_other - 1

        # Perform convolution
        convolved_signal = np.convolve(primary_signal, other_signal)


        convolved_signalsA.append(convolved_signal)

    print(convolved_signalsA)


    return convolved_signalsA


def convolveB():
    primary_file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    with open(primary_file_path, 'r') as file:
        primary_signal = np.loadtxt(file)[:, 1]

    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])

    convolved_signalsB = []

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            other_signal = np.loadtxt(file)

        # Calculate the length of the convolved signal
        len_primary = len(primary_signal)
        len_other = len(other_signal)
        len_convolved = len_primary + len_other - 1

        # Perform convolution
        convolved_signal = np.convolve(primary_signal, other_signal)

        convolved_signalsB.append(convolved_signal)

    print(convolved_signalsB)

    return convolved_signalsB

def Correlation2():
    X1 = signals[0]
    X2 = signals[1]
    N = len(signals[0])
    Cor = [0]*N
    for i in range (N):
        currentShift = shift_left(X2,i)
        for j in range (N):
            Cor[i] = Cor[i] + (X1[j]*currentShift[j])

        Cor[i] = Cor[i] * 1/N
# for signal in x
#     for i in signal
#x = convolve2()

#///////////////////////////////////////////////////////////////////////////////////// FIR
def FIR ():
    Filters = Filter.get()
    frequencySampling = float(Freq.get())
    StopBand = int(stopband.get())
    FC = float(fc.get())
    TransitionBand = float(Transband.get())
    PassBand = Passband.get()
    f1 =float(F1.get())
    f2 = float(F2.get())

    #first step -- window method
    if 44 < StopBand <= 53:   #hamming
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

    elif 53 <= StopBand <= 74: #blackman
        delta_f = TransitionBand / frequencySampling
        N = 5.5 / delta_f
        if N % 2 == 0:
            N += 1
        else:
            N = round(N)
            if N % 2 == 0:
                N += 1
        n = np.arange(-int((N - 1) / 2), int((N - 1) / 2) + 1)
        print ("the N:",N)
        print ("the range:",n)
        cos = np.cos(2*np.pi*n/(N-1))
        cos2 = np.cos(4*np.pi*n/(N-1))
        window_function = 0.42 + 0.5 *  cos + 0.08 * cos2
        print("the window function:",window_function)
        print("blackman")

    elif 21 <= StopBand <= 44 : #hanning
        delta_f = TransitionBand / frequencySampling
        print(delta_f)
        N = 3.1 / delta_f
        if N % 2 == 0:
            N += 1
        else:
            N = round(N)
            if N % 2 == 0:
                N += 1
        n = np.arange(-int((N - 1) / 2), int((N - 1) / 2) + 1)
        print("the N:", N)
        print("the range:", n)
        window_function = 0.5 + 0.5 * np.cos(2 * np.pi * n / N)
        print("the window function:",window_function)
        print("hanning")

    elif 1 <= StopBand <= 21 : #rectangular
        delta_f = TransitionBand / frequencySampling
        print(delta_f)
        N = 0.9 / delta_f
        if N % 2 == 0:
            N += 1
        else:
            N = round(N)
            if N % 2 == 0:
                N += 1
        n = np.arange(-int((N - 1) / 2), int((N - 1) / 2) + 1)
        print("the N:", N)
        print("the range:", n)
        window_function = 1
        print("the window function:",window_function)
        print("rectangular")

#second step --- filter
    if Filters == "low pass filter":
        HD = np.zeros_like(n, dtype=float)
        fc_new = (FC + (TransitionBand / 2)) / frequencySampling
        angular_fc = 2 * np.pi * fc_new
        xlen = len(n) // 2
        HD[0 + xlen] = 2 * fc_new

        for i, n_value in enumerate(n):
            if n_value != 0:
                HD[i] = (2 * fc_new * np.sin(n_value * angular_fc)) / (n_value * angular_fc)

        final_result = HD * window_function

        np.savetxt("I:/semester1materials/digital signal processing/outputfilter.txt", np.column_stack((n, final_result)), fmt="%d %.10f")

        Compare_Signals("I:/semester1materials/digital signal processing/Testcase 1/Testcase 1/LPFCoefficients.txt", n, final_result)

        for i, value in enumerate(final_result):
            print(f"{i - len(final_result) // 2} {value:.10f}")


    elif Filters == "high pass filter":
        HD = np.zeros_like(n, dtype=float)
        fc_new = (FC - (TransitionBand / 2)) / frequencySampling
        angular_fc = 2 * np.pi * fc_new
        xlen = len(n) // 2

        HD[0 + xlen] = 1 - (2 * fc_new)

        for i, n_value in enumerate(n):
            if n_value != 0:
                HD[i] = (-2 * fc_new) * (np.sin(n_value * angular_fc)) / (n_value * angular_fc)

        final_result = HD * window_function
        np.savetxt("I:/semester1materials/digital signal processing/outputfilter.txt", np.column_stack((n, final_result)), fmt="%d %.10f")

        Compare_Signals("I:/semester1materials/digital signal processing/Testcase 3/Testcase 3/HPFCoefficients.txt", n, final_result)

        for i, value in enumerate(final_result):
            print(f"{i - len(final_result) // 2} {value:.10f}")


    elif Filters == "Bandpass":
        HD = np.zeros_like(n, dtype=float)
        fc_new1 = (f1 - (TransitionBand / 2)) / frequencySampling
        fc_new2 = (f2 + (TransitionBand / 2)) / frequencySampling
        angular_fc1 = 2 * np.pi * fc_new1
        angular_fc2 = 2 * np.pi * fc_new2
        xlen = len(n) // 2
        HD[0 + xlen] = 2 * (fc_new2 - fc_new1)

        for i, n_value in enumerate(n):
            if n_value != 0:
                HD[i] = (2 * fc_new2 * ( np.sin(n_value * angular_fc2)/(n_value * angular_fc2) ) ) - (2 * fc_new1 * np.sin(n_value * angular_fc1)/(n_value * angular_fc1))


        final_result = HD * window_function
        np.savetxt("I:/semester1materials/digital signal processing/outputfilter.txt", np.column_stack((n, final_result)), fmt="%d %.10f")
        Compare_Signals("I:/semester1materials/digital signal processing/Testcase 5-20231220T210259Z-001/Testcase 5/BPFCoefficients.txt", n,final_result)

        for i, value in enumerate(final_result):
            print(f"{i - len(final_result) // 2} {value:.10f}")

    elif Filters == "Bandstop":
        HD = np.zeros_like(n, dtype=float)
        fc_new1 = (f1 + (TransitionBand / 2)) / frequencySampling
        fc_new2 = (f2 - (TransitionBand / 2)) / frequencySampling
        angular_fc1 = 2 * np.pi * fc_new1
        angular_fc2 = 2 * np.pi * fc_new2
        xlen = len(n) // 2
        HD[0 + xlen] = 1- (2 * (fc_new2 - fc_new1))

        for i, n_value in enumerate(n):
            if n_value != 0:
                HD[i] = (2 * fc_new1 * (np.sin(n_value * angular_fc1) / (n_value * angular_fc1))) - (2 * fc_new2 * np.sin(n_value * angular_fc2) / (n_value * angular_fc2))

        final_result = HD * window_function
        np.savetxt("I:/semester1materials/digital signal processing/outputfilter.txt", np.column_stack((n, final_result)), fmt="%d %.10f")

        Compare_Signals( "I:/semester1materials/digital signal processing/Testcase 7-20231220T210313Z-001/Testcase 7/BSFCoefficients/.txt", n, final_result)

        for i, value in enumerate(final_result):
            print(f"{i - len(final_result) // 2} {value:.10f}")

def consampling( signaloneData):
    signaltwoData, signaltwoindex = FIR()
    len_signal1 = len(signaloneData)
    len_signal2 = len(signaltwoData)
    len_convolved = len_signal1 + len_signal2 - 1

    signaloneData = np.pad(signaloneData, (0, len_convolved - len_signal1))
    signaltwoData = np.pad(signaltwoData, (0, len_convolved - len_signal2))

    fftone = fft(signaloneData)
    ffttwo = fft(signaltwoData)
    fftres = fftone * ffttwo
    convu = ifft(fftres).real

    return convu

def fast_conv(signaltwoindex, signaltwoData):
    # Initialize variables outside the if blocks
    signaloneData = 0


    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])

    if len(file_paths) == 1:
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                _signal_One = np.loadtxt(file, skiprows=rows_to_skip)
                signaloneData = list(zip(*_signal_One))[1]
                signaloneData = np.array(signaloneData)

                signaloneindex = list(zip(*_signal_One))[0]
                signaloneindex = np.array(signaloneindex)



    len_signal1 = len(signaloneData)
    len_signal2 = len(signaltwoData)
    len_convolved = len_signal1 + len_signal2 - 1

    signaloneData = np.pad(signaloneData, (0, len_convolved - len_signal1))
    signaltwoData = np.pad(signaltwoData, (0, len_convolved - len_signal2))

    fftone = fft(signaloneData)
    ffttwo = fft(signaltwoData)
    fftres = fftone * ffttwo
    ffinv = ifft(fftres).real

    common_values  = np.intersect1d(signaloneindex, signaltwoindex)
    array1 = np.setdiff1d(signaloneindex, common_values)
    array2 = np.setdiff1d(signaltwoindex, common_values)
    diff  = np.arange(400, 426)
    indexxx = np.concatenate([array2, array1, diff])
    # indexxx = np.append(signaloneindex, signaltwoindex)
    # indexxx = np.append(indexxx, indexxx[-1] + 1)
    # print(indexxx)
    # ConvTest(indexxx, ffinv)
    return indexxx, ffinv
def apply_fir():
    fir_signal, fir_index = FIR()
    fir_signal = fir_signal.astype(float)
    filtered_index , filtered_signal,  = fast_conv(fir_index,fir_signal)
    # filtered_index = filtered_index.astype(int)
    _filtered_index = np.arange(-26, 426)
    # Compare_Signals("E:/dsp/Practical task 1/FIR test cases/Testcase 2/ecg_low_pass_filtered.txt", _filtered_index, filtered_signal)



    return _filtered_index, filtered_signal

def resampling():

    M = int(decimation.get())
    L = int(interpolation.get())
    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])

    if len(file_paths) == 1:
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                original_signal = np.loadtxt(file, skiprows=rows_to_skip)[:, 1:]


    if M == 0 and L != 0:
        upsampled_signal = np.zeros((L - 1) * len(original_signal) + len(original_signal))
        upsampled_signal[::L] = original_signal.flatten()
        upsampled_signal = consampling(upsampled_signal)
        result_signal = upsampled_signal[:-2]
        _sampled_index = np.arange(-26, 1224)
        Compare_Signals("I:/semester1materials/digital signal processing/Testcase 2-resample/Testcase 2/Sampling_Up.txt", _sampled_index, result_signal)



    elif M != 0 and L == 0:
        dindex, downsampled_signal = apply_fir()
        downsampled_signal = downsampled_signal[::M]
        _sampled_index = np.arange(-26, 200)
        result_signal = downsampled_signal
        Compare_Signals("I:/semester1materials/digital signal processing/Testcase 1-resample/Testcase 1/Sampling_Down.txt", _sampled_index, result_signal)

    elif M != 0 and L != 0:
        upsampled_signal = np.zeros((L - 1) * len(original_signal) + len(original_signal))
        upsampled_signal[::L] = original_signal.flatten()
        upsampled_signal = consampling(upsampled_signal)
        todown_sampling = upsampled_signal[:-2]
        down_sampling = todown_sampling[::M]
        _sampled_index3 = np.arange(-26, 599)
        result_signal3 = down_sampling
        Compare_Signals("I:/semester1materials/digital signal processing/Testcase 3-resample/Testcase 3/Sampling_Up_Down.txt", _sampled_index3, result_signal3)


    else:
        print("Error: resampling failed")
        return

subjecta = []
subjectb = []
test = []
def read_subject_a():
    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])
    global subjecta


    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r') as file:
            signal = np.loadtxt(file, )
            subjecta.append(signal)

def read_subject_b():
    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])
    global subjectb


    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r') as file:
            signal = np.loadtxt(file, )
            subjectb.append(signal)

def read_test():
    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])
    global test


    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r') as file:
            signal = np.loadtxt(file, )
            test.append(signal)

def run_task2():
    global subjecta
    global subjectb
    global test
    print(subjecta)
    taskECG.run_all(subjecta, subjectb, test)



# Creating three frames
frame1 = tk.Frame(root, width=200, height=200, bg="lightgrey")
frame2 = tk.Frame(root, width=200, height=200, bg="lightblue")
frame3 = tk.Frame(root, width=200, height=200, bg="lightgreen")
frame4 = tk.Frame(root, width=200, height=200, bg="red")
frame5 = tk.Frame(root, width=200, height=200, bg="Turquoise")
frame6 = tk.Frame(root, width=200, height=200, bg="purple")
frame7 = tk.Frame(root, width=200, height=200, bg="orange")
frame8 = tk.Frame(root, width=200, height=200, bg="pink")

# Placing frames in different columns using the grid layout
frame1.grid(row=0, column=0)
frame2.grid(row=0, column=1)
frame3.grid(row=0, column=2)
frame4.grid(row=1, column=0)
frame5.grid(row=1, column=1)
frame6.grid(row=1, column=2)
frame7.grid(row=1, column=3)
frame8.grid(row=0, column=3)
# ///////////////////////////frame 1

label = tk.Label(frame1, text="Signal Generation:", font=label_font, anchor="w", justify="left")
label.pack()

button = tk.Button(frame1, text="Open File", command=button_click)
button.pack()

soc_label = tk.Label(frame1, text="Choose your waveform (sin or cos):", anchor="w", justify="left")
soc_label.pack()
soc_entry = tk.Entry(frame1)
soc_entry.pack()

amp_label = tk.Label(frame1, text="Amplitude:", anchor="w", justify="left")
amp_label.pack()
amp_entry = tk.Entry(frame1)
amp_entry.pack()

analog_label = tk.Label(frame1, text="Analog Frequency (Hz):", anchor="w", justify="left")
analog_label.pack()
analog_entry = tk.Entry(frame1)
analog_entry.pack()

sampling_label = tk.Label(frame1, text="Sampling Frequency (Hz):", anchor="w", justify="left")
sampling_label.pack()
sampling_entry = tk.Entry(frame1)
sampling_entry.pack()

shift_label = tk.Label(frame1, text="Phase Shift (in radians):", anchor="w", justify="left")
shift_label.pack()
shift_entry = tk.Entry(frame1)
shift_entry.pack()

# ///////////////////////////////////////////////////////////////////////////////2

result_label = tk.Label(frame1, text=" ", anchor="w", justify="left")
result_label.pack()


add_button = tk.Button(frame2, text="Add Signals", command=add)
add_button.grid(row=row_counter, column=0, padx=pad_x, sticky="w")


button = tk.Button(frame2, text="Read Signals", command=read_files)
button.grid(row=row_counter, column=1, padx=pad_x, sticky="w")
row_counter += 1


sub_button = tk.Button(frame2, text="Subtract Signals", command=subtract)
sub_button.grid(row=row_counter, column=0, padx=pad_x, sticky="w")
row_counter += 1

mul = tk.Label(frame2, text="Signal Multiplication")
mul.grid(row=row_counter, column=0, padx=pad_x, sticky="w")

con3 = tk.Entry(frame2)
con3.grid(row=row_counter, column=1, padx=pad_x, sticky="w")
row_counter += 1


m_button = tk.Button(frame2, text="Multiply", command=multiplication)
m_button.grid(row=row_counter, column=0, padx=pad_x, sticky="w")
row_counter += 1


sub_button = tk.Button(frame2, text="Square signals", command=squaring)
sub_button.grid(row=row_counter, column=0, padx=pad_x, sticky="w")
row_counter += 1

nor = tk.Label(frame2, text="Type (1) for (-1 to 1) or Type (2) for (0 to 1):")
nor.grid(row=row_counter, column=0, padx=pad_x, sticky="w")

normval = tk.Entry(frame2)
normval.grid(row=row_counter, column=1, padx=pad_x, sticky="w")
row_counter += 1


display_normalized_button = tk.Button(frame2, text="Display Normalized Signal", command=normalization)
display_normalized_button.grid(row=row_counter, column=0, padx=pad_x, sticky="w")
row_counter += 1

shift_label = tk.Label(frame2, text="Shift by (in samples):")
shift_label.grid(row=row_counter, column=0, padx=pad_x, sticky="w")

shift_value = tk.Entry(frame2)
shift_value.grid(row=row_counter, column=1, padx=pad_x, sticky="w")
row_counter += 1


shift_button = tk.Button(frame2, text="Shift Signal", command=shift)
shift_button.grid(row=row_counter, column=0, padx=pad_x, sticky="w")
row_counter += 1

acc_button = tk.Button(frame2, text="Accumulate Signal", command=acc)
acc_button.grid(row=row_counter, column=0, padx=pad_x, sticky="w")
row_counter += 1


# ////////////////////////////////////////////////////////////3

submit_button = tk.Button(frame3, text="Submit", command=submit)
submit_button.grid(row=row_counter, column=0, padx=pad_x, pady=10, sticky="w")
row_counter += 1


quantize_frame = tk.Frame(frame3)
quantize_frame.grid(row=row_counter, column=1, padx=pad_x, pady=10, sticky="w")

quantize_label = tk.Label(frame3, text="Quantize Signal")
quantize_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

submit_button1 = tk.Button(frame3, text="Display One", command=display)
submit_button1.grid(row=row_counter, column=0, padx=pad_x, pady=10, sticky="w")

submit_button2 = tk.Button(frame3, text="Display Two", command=display2)
submit_button2.grid(row=row_counter, column=1, padx=pad_x, pady=10, sticky="w")

row_counter += 1

# text_area = tk.Text(root, height=20, width=100)
# text_area.grid()

frame_letter = tk.Frame(frame3, padx=5, pady=5)
frame_letter.grid(row=row_counter, column=0, padx=10)
label_letter = tk.Label(frame_letter, text="Choose B or L:")
label_letter.grid(row=0, column=0, pady=5)
entry_letter = tk.Entry(frame_letter)
entry_letter.grid(row=1, column=0, pady=5)

frame_number = tk.Frame(frame3, padx=5, pady=5)
frame_number.grid(row=row_counter, column=1, padx=10)
label_number = tk.Label(frame_number, text="Choose Number:")
label_number.grid(row=0, column=0, pady=5)
entry_number = tk.Entry(frame_number)
entry_number.grid(row=1, column=0, pady=5)
#//////////////////////////////////////////////// 4

quantize_label = tk.Label(frame4, text="Frequency domain")
quantize_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

button = tk.Button(frame4, text="Read Signals", command=read_files2)
button.grid(row=1, column=0, padx=pad_x, sticky="w")
row_counter += 1
label_number = tk.Label(frame4, text="Choose Frequency:")
label_number.grid(row=2, column=0, pady=5)
con4 = tk.Entry(frame4)
con4.grid(row=2, column=1, padx=pad_x, sticky="w")
row_counter += 1

button4 = tk.Button(frame4, text="Display DFT", command=Fourier)
button4.grid(row=3, column=1, padx=pad_x, pady=10, sticky="w")
row_counter += 1

button4 = tk.Button(frame4, text="Display IDFT", command=Invers_Fourier)
button4.grid(row=3, column=0, padx=pad_x, pady=10, sticky="w")

label_amplitude = tk.Label(frame4, text="Enter new Amplitude:")
label_amplitude.grid(row=row_counter, column=1, padx=pad_x, pady=10, sticky="w")
row_counter += 1

entry_amplitude = tk.Entry(frame4)
entry_amplitude.grid(row=row_counter, column=1, padx=pad_x, pady=10, sticky="w")
row_counter += 1

label_phase = tk.Label(frame4, text="Enter new Phase in radians:")
label_phase.grid(row=row_counter, column=1, padx=pad_x, pady=10, sticky="w")
row_counter += 1

entry_phase = tk.Entry(frame4)
entry_phase.grid(row=row_counter, column=1, padx=pad_x, pady=10, sticky="w")
row_counter += 1

button_modify = tk.Button(frame4, text="Modify", command=modify_signal)
button_modify.grid(row=row_counter, column=1, padx=pad_x, pady=10, sticky="w")
row_counter += 1
#///////////////////////////////////////////////////////////////////////////////////// task 6
# Label and Entry for Filter Size
label_avg = tk.Label(frame5, text="Choose Filter Size:")
label_avg.grid(row=row_counter, column=0, padx=10, pady=10, sticky="w")

entry_avg = tk.Entry(frame5)
entry_avg.grid(row=row_counter, column=1, padx=10, pady=10, sticky="w")
row_counter += 1

# Buttons for Smoothing and Sharpening
smoothing_button = tk.Button(frame5, text="Smooth", command=smoothing)
smoothing_button.grid(row=row_counter, column=0, padx=10, pady=10, sticky="w")

sharpen_button = tk.Button(frame5, text="Sharpen", command=DerivativeSignal)
sharpen_button.grid(row=row_counter, column=1, padx=10, pady=10, sticky="w")
row_counter += 1

# Label and Entry for "Choose K"
label_k = tk.Label(frame5, text="Choose K:")
label_k.grid(row=row_counter, column=0, padx=10, pady=10, sticky="w")

entry_k = tk.Entry(frame5)
entry_k.grid(row=row_counter, column=1, padx=10, pady=10, sticky="w")
row_counter += 1

# Button for Folding
fold_button = tk.Button(frame5, text="Folding", command=folding)
fold_button.grid(row=row_counter, column=0, padx=10, pady=10, sticky="w")


# Buttons for Delaying and Advancing
sh_button = tk.Button(frame5, text="Shift fold", command=shiftFold)
sh_button.grid(row=row_counter, column=1, padx=10, pady=10, sticky="w")
row_counter += 1

# Buttons for Removing DC and Convolution
remove_dc_button = tk.Button(frame5, text="Remove DC", command=remove_f_dc)
remove_dc_button.grid(row=row_counter, column=0, padx=10, pady=10, sticky="w")

convolution_button = tk.Button(frame5, text="Convolute", command=convolve)
convolution_button.grid(row=row_counter, column=1, padx=10, pady=10, sticky="w")

#//////////////////////////////////////////////////////////////////// task 7

quantize_label = tk.Label(frame6, text="Correlation")
quantize_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
row_counter += 1

button = tk.Button(frame6, text="Read Signals", command=read_files)
button.grid(row=1, column=0, padx=pad_x, sticky="w")
row_counter += 1

button = tk.Button(frame6, text="Norm-Correlate", command=Correlation)
button.grid(row=1, column=1, padx=pad_x, sticky="w")
row_counter += 1

button = tk.Button(frame6, text="the time delay", command=Time_analysis)
button.grid(row=3, column=1, padx=pad_x, sticky="w")
row_counter += 1

label_number = tk.Label(frame6, text="Enter FS:")
label_number.grid(row=2, column=0, pady=5)
con5 = tk.Entry(frame6)
con5.grid(row=2, column=1, padx=pad_x, sticky="w")
row_counter += 1

button = tk.Button(frame6, text="Read Class1", command=read_files3)
button.grid(row=4, column=0, padx=pad_x, sticky="w")
row_counter += 1

button = tk.Button(frame6, text="average the class1 ", command=template_matching)
button.grid(row=4, column=1, padx=pad_x, sticky="w")
row_counter += 1

button = tk.Button(frame6, text="Read Class2", command=read_files3)
button.grid(row=5, column=0, padx=pad_x, sticky="w")
row_counter += 1

button = tk.Button(frame6, text="average the class2 ", command=template_matching2)
button.grid(row=5, column=1, padx=pad_x, sticky="w")
row_counter += 1


button = tk.Button(frame6, text="Read test", command=read_files3)
button.grid(row=6, column=0, padx=pad_x, sticky="w")
row_counter += 1

button = tk.Button(frame6, text="Compare", command=CorWithTest)
button.grid(row=6, column=1, padx=pad_x, sticky="w")
row_counter += 1

#//////////////////////////////////////////////////////////////////// FIR
label = tk.Label(frame7, text="FIR")
label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
row_counter += 1

label_fs= tk.Label(frame7, text="Enter Type of Filter:")
label_fs.grid(row=1, column=0, pady=5)
Filter = tk.Entry(frame7)
Filter.grid(row=1, column=1, padx=pad_x, sticky="w")

label_fs= tk.Label(frame7, text="Enter FS:")
label_fs.grid(row=2, column=0, pady=5)
Freq = tk.Entry(frame7)
Freq.grid(row=2, column=1, padx=pad_x, sticky="w")

label_number = tk.Label(frame7, text="Enter StopBandAttenuation:")
label_number.grid(row=3, column=0, pady=5)
stopband = tk.Entry(frame7)
stopband.grid(row=3, column=1, padx=pad_x, sticky="w")

label_number = tk.Label(frame7, text="Enter PassBandRipple:")
label_number.grid(row=4, column=0, pady=5)
Passband = tk.Entry(frame7)
Passband.grid(row=4, column=1, padx=pad_x, sticky="w")

label_number = tk.Label(frame7, text="Enter FC:")
label_number.grid(row=5, column=0, pady=5)
fc = tk.Entry(frame7)
fc.grid(row=5, column=1, padx=pad_x, sticky="w")

label_number = tk.Label(frame7, text="in case of (bandpass or stopband Enter F1:")
label_number.grid(row=6, column=0, pady=5)
F1 = tk.Entry(frame7)
F1.grid(row=6, column=1, padx=pad_x, sticky="w")

label_number = tk.Label(frame7, text="in case of (bandpass or stopband Enter F2:")
label_number.grid(row=7, column=0, pady=5)
F2 = tk.Entry(frame7)
F2.grid(row=7 ,column=1, padx=pad_x, sticky="w")

label_number = tk.Label(frame7, text="Enter TransitionBand:")
label_number.grid(row=8, column=0, pady=5)
Transband = tk.Entry(frame7)
Transband.grid(row=8, column=1, padx=pad_x, sticky="w")

button = tk.Button(frame7, text="Generate Filter (fir)", command=FIR)
button.grid(row=9, column=3, padx=pad_x, sticky="w")
row_counter += 1
button = tk.Button(frame7, text="Convolve", command=convolve)
button.grid(row=9, column=2, padx=pad_x, sticky="w")
row_counter += 1

label_number = tk.Label(frame7, text="Enter M:")
label_number.grid(row=9, column=0, pady=5)
decimation = tk.Entry(frame7)
decimation.grid(row=9, column=1, padx=pad_x, sticky="w")

label_number = tk.Label(frame7, text="Enter L:")
label_number.grid(row=10, column=0, pady=5)
interpolation = tk.Entry(frame7)
interpolation.grid(row=10, column=1, padx=pad_x, sticky="w")

# button = tk.Button(frame7, text="Generate Filter (fir)", command=FIR)
# button.grid(row=11, column=3, padx=pad_x, sticky="w")
#
# filteredButton = tk.Button(frame7, text="Apply Filter", command=apply_fir)
# filteredButton.grid(row=12, column=3, padx=pad_x, sticky="w")

resampleButton = tk.Button(frame7, text="resample", command=resampling)
resampleButton.grid(row=13, column=3, padx=pad_x, sticky="w")


#////////////////////////////////////////////////////////////////////////////////////////////
label = tk.Label(frame8, text="ECG")
label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
row_counter += 1

button = tk.Button(frame8, text="read subject A ", command=read_subject_a)
button.grid(row=1, column=3, padx=pad_x, sticky="w")

button = tk.Button(frame8, text="read subject B", command=read_subject_b)
button.grid(row=2, column=3, padx=pad_x, sticky="w")

button = tk.Button(frame8, text="read test", command=read_test)
button.grid(row=3, column=3, padx=pad_x, sticky="w")

button = tk.Button(frame8, text="run", command=run_task2)
button.grid(row=4, column=3, padx=pad_x, sticky="w")









root.mainloop()
root.mainloop()