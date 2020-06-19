import numpy as np
import matplotlib.pyplot as plt

# pad seismic data from different composite trace to make them have the same length
def data_pad(data_in, t_min, t_max, t0, td, dt):
    Nt = int((td - t0) / dt + 1)
    data_tmp = np.zeros((Nt, data_in.shape[1]))
    nt1 = 0
    for nt in range(Nt):
        tt = t0 + nt * dt
        #print(tt)
        if tt >= t_min and tt <= t_max:
            data_tmp[nt, :] = data_in[nt1, :]
            nt1 += 1
    return data_tmp

def compLoad(path_basis, path_data, well, Nsk, t_min, t_max, dt=2.0):
    file_near = '\composite_trace_' + well + '_near.txt'  # make sure the file name complies to name rountine
    file_mid = '\composite_trace_' + well + '_mid.txt'
    file_far = '\composite_trace_' + well + '_far.txt'
    seis_near = np.asarray(np.loadtxt(path_basis + path_data + file_near, skiprows=Nsk))
    seis_mid = np.asarray(np.loadtxt(path_basis + path_data + file_mid, skiprows=Nsk))
    seis_far = np.asarray(np.loadtxt(path_basis + path_data + file_far, skiprows=Nsk))

    Nt_min = int(t_min / dt)
    Nt_max = int(t_max / dt) + 1  # make sure to include the point at t_max, so we need to add 1 here.

    # Find the target interval for seismic data
    t_axis = seis_near[Nt_min:Nt_max, 0]
    seis_near = seis_near[Nt_min:Nt_max, 1]
    seis_mid = seis_mid[Nt_min:Nt_max, 1]
    seis_far = seis_far[Nt_min:Nt_max, 1]


    seis_gather = np.transpose(np.stack((t_axis, seis_near, seis_mid, seis_far), axis=0))

    return seis_gather

def Reflectivity(VP, VS, DEN, step, angle_max):
    num = len(VP)
    angle = np.arange(0, angle_max, step)
    #print(angle, num)
    RC = np.zeros((num-1, int(len(angle))))
    #print(RC.shape)
    pi = 3.1415926
    for i in range(num-1):
        vp_avg = (VP[i] + VP[i + 1]) / 2.0
        vs_avg = (VS[i] + VS[i + 1]) / 2.0
        den_avg = (DEN[i] + DEN[i + 1]) / 2.0
        vp_diff = VP[i + 1] - VP[i]
        vs_diff = VS[i + 1] - VS[i]
        den_diff = DEN[i + 1] - DEN[i]
        for j in range(int(len(angle))):
            ang = j * step
            angr = ang * pi / 180
            angle2 = np.arcsin(VP[i+1] / VP[i] * np.sin(angr))
            angle_avg = (angle2 + angr) / 2.0
            RC[i, j] = 1.0/2.0 * (1 - 4.0 * vs_avg**2.0 / (vp_avg * vp_avg) * np.sin(angle_avg)**2.0) * den_diff / den_avg
        + 1.0 / (2.0 * np.cos(angle_avg)**2.0) * (vp_diff/vp_avg) - 4.0 * vs_avg * vs_avg / (vp_avg * vp_avg) * (vs_diff / vs_avg) * (np.sin(angle_avg)**2.0)

    return RC

def sig_add_noise(signal, snr_des):
    Ns = len(signal)
    noise = np.random.normal(0, 1, Ns)
    sig_pow = sum(np.multiply(signal, signal)) / Ns
    noise_pow = sum(np.multiply(noise, noise)) / Ns
    #initial_SNR = 10 * np.log10(sig_pow / noise_pow)
    #print(sig_pow, noise_pow, initial_SNR)
    ks = (sig_pow / noise_pow) * pow(10, -snr_des / 10)  # scale_factor
    noise_new = np.sqrt(ks) * noise  # change the noise level
    #noise_pow_new = sum(np.multiply(noise_new, noise_new)) / Ns
    #snr_new = 10 * np.log10(sig_pow / noise_pow_new)
    sig_noise = signal + noise_new
    return sig_noise

import math
# Nt: the number of samples
# f: dominant frquency
# dt: sample rate (unit: ms)
# A: amplitude of ricker wavelet
def rickerlj(Nt,f,dt,A):
    taxis = np.zeros(Nt)
    tw = np.zeros(Nt)
    ricker=np.zeros(Nt)
    for i in range(Nt):
        taxis[i] = i * dt
        tw[i] = taxis[i] * 0.001 - int(np.ceil(float(Nt / 2 - 1))) * dt * 0.001
        ricker[i] =A*(1 - (2 * np.pi ** 2) * (f ** 2) * (tw[i] ** 2)) * math.exp((-np.pi ** 2) * (f ** 2) * (tw[i] ** 2))
    tw = tw * 1000 # units change from s to ms.
    return ricker, tw

def syn_data(data_well, table, t_min, t_max, dt):
    # depth-time conversion
    #coef = linfit(table, dt, t_min, t_max)
    #time_seis = data_well[:, 0] * coef[0] + coef[1]  # The t-d function is calculated from well tie for that well
    VP = td_convert_v1(data_well[:, 7], data_well[:, 0], table, dt, t_min, t_max)
    VS = td_convert_v1(data_well[:, 8], data_well[:, 0], table, dt, t_min, t_max)
    DEN = td_convert_v1(data_well[:, 9], data_well[:, 0], table, dt, t_min, t_max)
    #print(VP.shape, VS.shape, DEN.shape)

    angle_max0 = 30  # maximum reflection angle
    ang_step = 1  # increment of reflection angle
    RC = Reflectivity(VP, VS, DEN, ang_step, angle_max=angle_max0)

    # preprocessing on reflectivity series, remove the abnormal value
    min = -0.8
    max = 0.8
    Num, Nag = RC.shape
    for i in range(Num):
        for j in range(Nag):
            if RC[i, j] < min:
                RC[i, j] = min
            if RC[i, j] > max:
                RC[i, j] = max

    # ricker wavelet parameter:
    Nt = 60
    f = 30  # unit: Hz
    A = 10.0
    ricker, t_axis = rickerlj(Nt, f, dt, A)
    #print("ricker size", len(ricker))

    fs = int(1 / dt * 1000)  # sampling frequency, Hz
    N_f = int(fs)  # number of FFT Point, greater or equal to the sample number of hanning window?
    df = 1.0
    #seis_f = np.abs(np.fft.fft(ricker, n=N_f))
    """seis_f = np.abs(np.fft.fft(ricker, n=N_f))
    SF = np.fft.fft(ricker, n=N_f)
    seis_t = np.real(np.fft.ifft(SF, n=N_f))

    print(seis_t.shape, seis_t)


    plt.figure(14)
    plt.plot(t_axis, ricker)

    plt.figure(15)
    plt.plot(seis_f)
    #plt.xlim(0, N_f)
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("Amplitude")

    plt.figure(16)
    plt.plot(seis_t[0: 60])
    #plt.xlim(0, N_f)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    #plt.show()

    xt_r = np.zeros((Nt))
    taxis_r = np.zeros((Nt))
    Xw = SF
    pi=3.1415926
    T_window = Nt
    for nt in range(int(Nt)):
        ts = (nt) * dt  # need to be very careful!!
        taxis_r[nt] = ts
        xt_all = 0.0
        for nw in range(int(100)):
            f = nw * df
            xt_tmp = np.real(Xw[nw]) * np.cos(2 * pi * f * ts * 0.001) - np.imag(Xw[nw]) * np.sin(2 * pi * f * ts * 0.001)
            xt_all += xt_tmp
        xt_r[nt] = xt_all / fs *2

    t_axis = np.linspace(0, T_window * 2 - 2.0, num=int(Nt))
    # print(t_axis, taxis_r)
    plt.figure(17)
    plt.plot(taxis_r, xt_r, '-', color='r')
    plt.plot(t_axis, ricker, '-', color='black')
    # plt.xlim(0, 1000)
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude")
    plt.show()"""

    seis_syn = []
    for ntr in range(Nag):
        syn = np.convolve(RC[:, ntr], ricker, 'same')
        # print(syn.shape, type(syn))
        seis_syn.append(syn)

    seis_syn = np.array(seis_syn)

    #print("seis_syn size:", seis_syn.shape)
    ang_size = 10
    Ng = int(Nag * ang_step / ang_size)  # set up the number of stack seismic data

    seis_syn_sum = []
    for i in range(Ng):
        seis_syn_tmp2 = 0
        for j in range(int(ang_size / ang_step)):
            seis_syn_tmp2 += seis_syn[int(i * ang_size / ang_step) + j, :]
            # print(i*ang_size+j)
        seis_syn_sum.append(seis_syn_tmp2)

    seis_syn_sum = np.array(seis_syn_sum)
    #print(seis_syn_sum.shape)

    #np.save("Seis_syn_gather", seis_syn_sum)

    return seis_syn_sum


# convert to time domain, convolve with wavelet, and sampling the result with dt
import scipy
from scipy import fftpack
def syn_data_v1(data_well, table, t_min, t_max, dt):
    t_min = t_min - 2.0
    dt_tmp = 0.5
    VP = td_convert_v1(data_well[:, 7], data_well[:, 0], table, dt_tmp, t_min, t_max)
    VS = td_convert_v1(data_well[:, 8], data_well[:, 0], table, dt_tmp, t_min, t_max)
    DEN = td_convert_v1(data_well[:, 9], data_well[:, 0], table, dt_tmp, t_min, t_max)
    print(VP.shape, VS.shape, DEN.shape)

    angle_max0 = 30  # maximum reflection angle
    ang_step = 1  # increment of reflection angle
    RC = Reflectivity(VP, VS, DEN, ang_step, angle_max=angle_max0)

    # preprocessing on reflectivity series, remove the abnormal value
    min = -0.8
    max = 0.8
    Num, Nag = RC.shape
    for i in range(Num):
        for j in range(Nag):
            if RC[i, j] < min:
                RC[i, j] = min
            if RC[i, j] > max:
                RC[i, j] = max

    # ricker wavelet parameter:
    Nt = 240
    f = 30  # unit: Hz
    A = 1000.0
    dtw = 0.5  # sampling itnerval for wavelet
    ricker, t_axis = rickerlj(Nt, f, dtw, A)
    print("ricker size", len(ricker))

    fs = int(1 / dtw * 1000) / 4  # sampling frequency, Hz
    N_f = int(fs * 2)  # number of FFT Point, greater or equal to the sample number of hanning window?
    #seis_f = np.abs(np.fft.fft(ricker))
    seis_f = np.abs(scipy.fftpack.fft(ricker))
    df = 1 / (Nt * dtw * 0.001)  # this is how to calculate df.
    #print(seis_f.shape)
    #F_axis = np.arange(1, N_f+1, 1.0)

    """plt.figure(14)
    plt.plot(t_axis, ricker)

    plt.figure(15)
    plt.plot(seis_f)
    #plt.xlim(0, N_f)
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("Amplitude")
    plt.show()"""

    seis_syn = []
    for ntr in range(Nag):
        syn = np.convolve(RC[:, ntr], ricker, 'same')
        # print(syn.shape, type(syn))
        seis_syn.append(syn)

    seis_syn = np.array(seis_syn)

    print("seis_syn size:", seis_syn.shape)

    ang_size = 10
    Ng = int(Nag * ang_step / ang_size)  # set up the number of stack seismic data

    seis_syn_sum = []
    RC_sum = []
    RC = np.transpose(RC)
    for i in range(Ng):
        seis_syn_tmp2 = 0
        rc_tmp2 = 0
        for j in range(int(ang_size / ang_step)):
            seis_syn_tmp2 += seis_syn[int(i * ang_size / ang_step) + j, :]
            rc_tmp2 += RC[int(i * ang_size / ang_step) + j, :]
            # print(i*ang_size+j)
        RC_sum.append(rc_tmp2/ang_size )
            # print(i*ang_size+j)
        seis_syn_sum.append(seis_syn_tmp2)

    seis_syn_sum = np.asarray(seis_syn_sum)
    RC_sum = np.asarray(RC_sum)
    print('seis syn sum shape:', seis_syn_sum.shape)


    Ng, Nt = seis_syn_sum.shape
    seis_out = []
    RC_out = []
    for ng in range(Ng):
        seis_tmp = []
        RC_tmp = []
        #seis_tmp = td_convert_v1(seis_syn_sum[ng, :], data_well[:, 0], table, dt, t_min, t_max)
        for nt in range(Nt):
            if nt % 4 == 0:
                seis_tmp.append(seis_syn_sum[ng, nt])
                RC_tmp.append(RC_sum[ng, nt])
        seis_out.append(seis_tmp)
        RC_out.append(RC_tmp)

    seis_out = np.asarray(seis_out)
    RC_out = np.asarray(RC_out)
    #print(seis_out.shape)

    #np.save("Seis_syn_gather", seis_out)

    return seis_out, RC_out

from scipy import signal
def STFT(data_in, dt, T_window):
    data_in_cut = data_in  # for one diemensional data
    #print(data_in_cut.shape)
    Nt = len(data_in_cut)
    #print(Nt)

    dt = float(dt)
    fs = int(1 / dt * 1000) + 2 # sampling frequency, Hz
    N_f = int(fs)  # number of FFT Point, greater or equal to the sample number of hanning window?
    #print(N_f, fs, dt)
    #df = float(fs) / float(N_f)  # sampling frequency
    #fnyqst = int(np.ceil(float(fs / 2)))
    #print(fnyqst, df)
    T_window = int(round(T_window /dt))  # calculate the sampling number of T_window

    W_han1 = np.hanning(T_window + 1)
    W_han2 = W_han1[0:T_window]  # form periodic Hanning window, the reason need to be reevaluated. the erason might be, periodic
    # Hanning might be good for FFT/DFT
    #print(W_han2.shape)

    data_tmp = np.zeros((Nt + T_window), dtype=float)
    data_tmp[int(np.ceil(float(T_window / 2))):Nt + int(np.ceil(float(T_window / 2)))] = data_in_cut

    coln = int(fs/2)  # the total number of output frequency, just half of total N_f

    #STFT_TFP_out = np.zeros((Nt, coln), dtype=complex)  # create the Time-frequency matrix
    STFT_out = np.zeros((Nt, coln), dtype=float)
    STFT_out_real = np.zeros((Nt, coln), dtype=float)
    STFT_out_img = np.zeros((Nt, coln), dtype=float)
    #print(STFT_TFP_out.shape)
    #print(int(np.ceil(float(T_window / 2))))

    # calculate the window length for savgo_filter, which must be a positive and odd number.
    len_s = T_window // 2
    if (len_s % 2) == 0:
        len_s = len_s - 1

    # perform STFT
    for t in range(int(np.ceil(float(T_window / 2))), Nt + int(np.ceil(float(T_window / 2)))):
        # add Han window on original data
        # Y = data_tmp[t-int(np.ceil(float(T_window/2))):T_window + t-int(np.ceil(float(T_window/2)))]
        # print(Y.shape)
        tmp = data_tmp[t - int(np.ceil(float(T_window / 2))):T_window + t - int(np.ceil(float(T_window / 2)))]

        tmp_s = signal.savgol_filter(tmp, len_s, 3)  # smooth the signal..
        A_han = max(tmp_s)
        if A_han == 0:
            A_han=max(data_in_cut)
        W_han = A_han * W_han2

        Xt = np.multiply(tmp, W_han) / A_han   # make sure the amplitude is the same after application of Hannning window.

        # Apply FFT to data
        Xw = np.fft.fft(Xt, N_f)
        #Xw = np.fft.fft(tmp_s, N_f)

        STFT_out[t - int(np.ceil(float(T_window / 2))), :] = np.abs(Xw[0:coln])
        STFT_out_img[t - int(np.ceil(float(T_window / 2))), :] = np.imag(Xw[0:coln])
        STFT_out_real[t - int(np.ceil(float(T_window / 2))), :] = np.real(Xw[0:coln])
    #print(STFT_TFP_out[150, 0:50])
    #print(STFT_out.shape)

    return STFT_out, STFT_out_real, STFT_out_img

def STFT_v1(data_in, dt, T_window):
    data_in_cut = data_in  # for one diemensional data
    #print(data_in_cut.shape)
    Nt = len(data_in_cut)
    #print(Nt)

    dt = float(dt)
    fs = int(1 / dt * 1000) + 2 # sampling frequency, Hz
    N_f = int(fs)  # number of FFT Point, greater or equal to the sample number of hanning window?
    #print(N_f, fs, dt)
    #df = float(fs) / float(N_f)  # sampling frequency
    #fnyqst = int(np.ceil(float(fs / 2)))
    #print(fnyqst, df)
    T_window = int(round(T_window /dt))  # calculate the sampling number of T_window

    W_han1 = np.hanning(T_window + 1)
    W_han2 = W_han1[0:T_window]  # form periodic Hanning window, the reason need to be reevaluated. the reason might be, periodic
    # Hanning might be good for FFT/DFT
    #print('W_hand2 shape is:', W_han2.shape, T_window)

    """plt.figure(1)
    plt.plot(W_han1, '-', color='black')
    plt.plot(W_han2, '-', color='r')
    # plt.xlim(0, 1000)
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("Amplitude")"""



    data_tmp = np.zeros((Nt + T_window), dtype=float)
    data_tmp[int(np.ceil(float(T_window / 2))):Nt + int(np.ceil(float(T_window / 2)))] = data_in_cut

    """plt.figure(2)
    plt.plot(data_tmp, '-', color='black')
    plt.plot(data_in_cut, '-', color='r')
    # plt.xlim(0, 1000)
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("Amplitude")
    plt.show()"""

    coln = int(fs/2)  # the total number of output frequency, just half of total N_f

    #STFT_TFP_out = np.zeros((Nt, coln), dtype=complex)  # create the Time-frequency matrix
    STFT_out = np.zeros((Nt, coln), dtype=float)
    STFT_out_real = np.zeros((Nt, coln), dtype=float)
    STFT_out_img = np.zeros((Nt, coln), dtype=float)
    #print(STFT_TFP_out.shape)
    #print(int(np.ceil(float(T_window / 2))))

    # calculate the window length for savgo_filter, which must be a positive and odd number.
    len_s = T_window // 2
    if (len_s % 2) == 0:
        len_s = len_s - 1

    # perform STFT
    for t in range(int(np.ceil(float(T_window / 2))), Nt + int(np.ceil(float(T_window / 2)))):
        # add Han window on original data
        # Y = data_tmp[t-int(np.ceil(float(T_window/2))):T_window + t-int(np.ceil(float(T_window/2)))]
        # print(Y.shape)
        tmp = data_tmp[t - int(np.ceil(float(T_window / 2))):T_window + t - int(np.ceil(float(T_window / 2)))]
        #print(t, t - int(np.ceil(float(T_window / 2))), T_window + t - int(np.ceil(float(T_window / 2))))

        tmp_s = signal.savgol_filter(tmp, len_s, 3)  # smooth the signal..



        A_han = max(tmp_s)
        if A_han == 0:
            A_han=max(data_in_cut)
        W_han = A_han * W_han2

        #Xt = np.multiply(tmp, W_han) / A_han   # make sure the amplitude is the same after application of Hannning window.
        Xt = tmp  # remove the effect of window.

        """if T_window == 80 and t > T_window + 20:
            plt.figure(3)
            plt.plot(tmp, '-', color='black')
            plt.plot(tmp_s, '-', color='r')
            # plt.xlim(0, 1000)
            plt.xlabel("Frequency(Hz)")
            plt.ylabel("Amplitude")

            plt.figure(4)
            plt.plot(tmp, '-', color='black')
            plt.plot(Xt, '-', color='r')
            plt.plot(W_han, '-', color='b')
            # plt.xlim(0, 1000)
            plt.xlabel("Frequency(Hz)")
            plt.ylabel("Amplitude")
            plt.show()"""


        # Apply FFT to data
        Xw = np.fft.fft(Xt, N_f)
        """pi = 3.1415926
        df = 1.0
        # how to recover the singal even with a small window.
        if T_window == 10 and t > T_window - 20:
            Nt = T_window
            #xt_rs = np.real(np.fft.ifft(Xw, int(N_f/2)))
            #print(xt_rs)
            xt_r = np.zeros((int(Nt)))
            taxis_r = np.zeros((int(Nt)))
            for nt in range(int(Nt)):
                ts = nt * dt  # need to be very careful!!
                taxis_r[nt] = ts
                xt_all = 0.0
                for nw in range(int(150)):
                    f = nw * df
                    xt_tmp = np.real(Xw[nw]) * np.cos(2 * pi * f * ts * 0.001) - np.imag(Xw[nw]) * np.sin(2 * pi * f * ts * 0.001)
                    xt_all += xt_tmp
                xt_r[nt] = xt_all / 252.5
            t_axis = np.linspace(0, T_window * 2 - 2.0, num=int(len(Xt)))
            # print(t_axis, taxis_r)
            plt.figure(5)
            plt.plot(taxis_r, xt_r, '-', color='r')
            plt.plot(t_axis, Xt, '-', color='black')
            # plt.xlim(0, 1000)
            plt.xlabel("Time(ms)")
            plt.ylabel("Amplitude")

            plt.show()"""

        #Xw = np.fft.fft(tmp_s, N_f)

        STFT_out[t - int(np.ceil(float(T_window / 2))), :] = np.abs(Xw[0:coln])
        STFT_out_img[t - int(np.ceil(float(T_window / 2))), :] = np.imag(Xw[0:coln])
        STFT_out_real[t - int(np.ceil(float(T_window / 2))), :] = np.real(Xw[0:coln])
    #print(STFT_TFP_out[150, 0:50])
    #print(STFT_out.shape)

    return STFT_out, STFT_out_real, STFT_out_img

def MSFT(seis_gather, dt, N_win, Fout, T0, ddt):
    # limiting the range for reducing the computing burden for testing of code
    seis_tmp = seis_gather[:]
    Nag = 1
    Ntr = len(seis_tmp)
    print("the length of final data used in calculation is:", len(seis_tmp))
    fs = int(1 / dt * 1000) # sampling frequency, Hz
    N_f = int(fs)  # number of FFT Point, greater or equal to the sample number of hanning window?
    df = float(fs) / float(N_f)  # Sampling frequency interval
    pi = 3.1415926

    NFout = int(Fout / df) + 1  # the number of output points
    Ns = 200 # Just an approximation, cause each composite trace has different length
    Memory_need = Ntr * NFout * N_win * Nag * Ns * 6 * 8 * 0.000000001
    print("the memory needed for data (GB):", Memory_need)

    MSFT_f_all = []
    MSFT_f_real_all = []
    MSFT_f_img_all = []
    MSFT_t_all = []
    MSFT_real_t_all = []
    MSFT_img_t_all = []

    print("Calculating MSFT:")
    numW = 1
    numT = len(seis_gather)
    for seis_tmp in seis_gather:
        print('Processing well ', numW, ' of', numT, 'wells\n')
        MSFT_f = []
        MSFT_f_real = []
        MSFT_f_img = []
        MSFT_t = []
        MSFT_real_t = []
        MSFT_img_t = []
        Nt = len(seis_tmp) # Nt is the sample number of each trace, Ntag is the dimension of angle gather. Here is about 100 x 3

        # Check the dimension of original seismic data
        if Nag == 3:
            seis_tmp_v1 = np.zeros((Nt, Nag+1))
            seis_tmp_v1[:, 1:] = seis_tmp
            seis_tmp = seis_tmp_v1
            Nag = 4
        else:
            Nag = 2
        #print(seis_tmp.shape)

        for ng in range(Nag-1):
            if Nag == 4:
                seis_tmp1 = seis_tmp[:, ng+1]  # the first column is time index
            else:
                seis_tmp1 = seis_tmp
            MSFT_f_tmp = []
            MSFT_f_real_tmp = []
            MSFT_f_img_tmp = []
            MSFT_t_tmp = []
            MSFT_real_t_tmp = []
            MSFT_img_t_tmp = []
            for i in range(N_win):
                T_win = T0 + i * ddt
                #print(seis_tmp1.shape, dt, T_win)
                stft_out_v1, stft_out_v2, stft_out_v3 = STFT_v1(seis_tmp1, dt, T_win)  # they are abs, real, and image part of Fourier transform
                #stft_out_v1, stft_out_v2, stft_out_v3 = STFT(seis_tmp1, dt, T_win)
                # calculate STFT in time domain with frequency domain data.
                Nwin = int(T_win / dt)

                stft_out_v1_t = np.zeros((Nt, NFout))
                stft_out_v2_t = np.zeros((Nt, NFout))
                stft_out_v3_t = np.zeros((Nt, NFout))

                for k in range(Nt):
                    for nnf in range(NFout):
                        nf = nnf * df
                        f1 = fs - nf
                        if nnf == 0.0:
                            xt_tmp1 = 0.0
                            xt_tmp2 = 0.0
                        else:
                            xt_tmp1 = stft_out_v2[k, nnf] * np.cos(2 * pi * f1 * (np.ceil(Nwin / 2)) * dt * 0.001)
                            xt_tmp2 = stft_out_v3[k, nnf] * np.sin(2 * pi * f1 * (np.ceil(Nwin / 2)) * 0.001)  # the imaginary part is odd function, so need to change sign here.
                        stft_out_v1_t[k, nnf] = stft_out_v2[k, nnf] * np.cos(2 * pi * nf * (np.ceil(Nwin / 2)) * dt * 0.001) - \
                                stft_out_v3[k, nnf] * np.sin(2 * pi * nf * (np.ceil(Nwin / 2)) * dt * 0.001) + xt_tmp1 + xt_tmp2
                        stft_out_v2_t[k, nnf] = stft_out_v2[k, nnf] * np.cos(2 * pi * nf * np.ceil(Nwin / 2) * dt * 0.001) + xt_tmp1
                        stft_out_v3_t[k, nnf] = -stft_out_v3[k, nnf] * np.sin(2 * pi * nf * np.ceil(Nwin / 2) * dt * 0.001) + xt_tmp2   # the imaginary part is zero
                        #stft_out_v2_t[k, nnf] = stft_out_v2[k, nnf] * np.cos(2 * pi * nf * np.ceil(Nwin / 2) * dt * 0.001)
                        #stft_out_v3_t[k, nnf] = -stft_out_v3[k, nnf] * np.sin(2 * pi * nf * np.ceil(Nwin / 2) * dt * 0.001)


                MSFT_f_tmp.append(stft_out_v1[:, 0:NFout])
                MSFT_f_real_tmp.append(stft_out_v2[:, 0:NFout])
                MSFT_f_img_tmp.append(stft_out_v3[:, 0:NFout])
                MSFT_t_tmp.append(stft_out_v1_t)
                MSFT_real_t_tmp.append(stft_out_v2_t)
                MSFT_img_t_tmp.append(stft_out_v3_t)

            # Append result for each angle trace:
            MSFT_f.append(MSFT_f_tmp)
            MSFT_f_real.append(MSFT_f_real_tmp)
            MSFT_f_img.append(MSFT_f_img_tmp)
            MSFT_t.append(MSFT_t_tmp)
            MSFT_real_t.append(MSFT_real_t_tmp)
            MSFT_img_t.append(MSFT_img_t_tmp)

        # Append each well composite trace:
        MSFT_f_all.append(MSFT_f)
        MSFT_f_real_all.append(MSFT_f_real)
        MSFT_f_img_all.append(MSFT_f_img)
        MSFT_t_all.append(MSFT_t)
        MSFT_real_t_all.append(MSFT_real_t)
        MSFT_img_t_all.append(MSFT_img_t)
        numW +=1

    # seis_syn_r = 1 / fs * np.sum(stft_out_v2_t[:, :], axis=1)

    #np.save("MSFT_f", MSFT_f_all)
    #np.save("MSFT_t", MSFT_t_all)
    #np.save("MSFT_t_real", MSFT_real_t_all)
    #np.save("MSFT_t_img", MSFT_img_t_all)

    return MSFT_f_all, MSFT_f_real_all, MSFT_f_img_all, MSFT_t_all, MSFT_real_t_all, MSFT_img_t_all

def DataPrepV1(MSFT, dt, f_intv, Fout):
    fs = int(1 / dt * 1000)  # sampling frequency, Hz
    N_f = int(fs * 2)  # number of FFT Point, greater or equal to the sample number of hanning window?
    df = fs / N_f

    # calculate the output of MSFT for the corresponding frequency band
    fmax = int(Fout / df)
    fmin = int((Fout - f_intv) / df)
    MSFT_out = np.sum(MSFT[:, fmin:fmax, :, :], axis=1)  # 10 to 20 Hz for all phase.

    return MSFT_out

def linfit(dt_table, dt, t_min, t_max):
    # Find the target interval for Time-Depth table
    Ntd = len(dt_table)
    t_max = 2710.0  # make sure use the right maximum value for linear fitting.
    ntd_min = 0
    ntd_max = 0
    for itd in range(Ntd):
        if dt_table[itd, 2] > t_min and dt_table[itd, 2] < t_min + dt + 0.5:
            ntd_min = itd

        if dt_table[itd, 2] > t_max and dt_table[itd, 2] < t_max + dt + 0.5:
            ntd_max = itd - 1

    #print(dt_table[ntd_min, 2], dt_table[ntd_max, 2])
    dtTable_edit = dt_table[ntd_min:ntd_max, :]
    #print(dtTable_edit)

    # Find the linear fit coefficient, make sure the time is the function of depth

    fit_coef = np.polyfit(dtTable_edit[:, 0], dtTable_edit[:, 2], 3)

    return fit_coef, dtTable_edit

def linfit_v1(dt_table, dt, t_min, t_max):
    # Find the target interval for Time-Depth table
    Ntd = len(dt_table)
    t_max = 2710.0  # make sure use the right maximum value for linear fitting.
    ntd_min = 0
    ntd_max = 0
    for itd in range(Ntd):
        if dt_table[itd, 2] > t_min and dt_table[itd, 2] < t_min + dt + 0.5:
            ntd_min = itd

        if dt_table[itd, 2] > t_max and dt_table[itd, 2] < t_max + dt + 0.5:
            ntd_max = itd - 1

    #print(dt_table[ntd_min, 2], dt_table[ntd_max, 2])
    dtTable_edit = dt_table[ntd_min:ntd_max, :]
    #print(dtTable_edit)

    # Find the linear fit coefficient, make sure the time is the function of depth

    fit_coef = np.polyfit(dtTable_edit[:, 0], dtTable_edit[:, 2], 1)

    return fit_coef

def td_convert(por_data, depth, dt_table, dt, t_min, t_max):
    # depth-time conversion
    idx = 0
    por_final = []
    #print(len(por_data))
    for por in por_data:
        print('processing well', idx+1)
        #print(len(por))
        dpt = depth[idx]
        table = dt_table[idx]
        t_min_tmp = t_min[idx]
        t_max_tmp = t_max[idx]
        coef, table_edit = linfit(table, dt, t_min_tmp, t_max_tmp)

        #print(dpt, coef)
        #time_seis = dpt**2.0 * coef[0] + dpt * coef[1] + coef[2] # different wells have different TD table
        #time_seis = table_edit[:, 0] ** 3.0 * coef[0]+ table_edit[:, 0] ** 2.0 * coef[1] + table_edit[:, 0] * coef[2] + coef[3]
        time_seis = dpt ** 3.0 * coef[0] + dpt ** 2.0 * coef[1] + dpt * coef[2] + coef[3]
        """f, bx = plt.subplots(nrows=1, ncols=1)
        bx.plot(table_edit[:, 2], '-', color='black')
        bx.plot(time_seis, '-', color='r')"""

        #plt.show()

        #print(time_seis)
        t_flag = t_min_tmp
        nsp = 0
        por_edit = []
        por_tmp = []
        for t_tmp in time_seis:
            #print(t_tmp)

            if t_tmp >= t_flag - dt/2.0 and t_tmp <= t_flag + dt/2.0:
                por_tmp.append(por[nsp])
                #print(por[np])

            if t_tmp > t_flag + dt/2.0:

                #print(len(por_tmp))
                #print(por_tmp)
                if len(por_tmp) > 1:
                    por_edit.append(sum(por_tmp)/len(por_tmp))  # Calculate the porosity use the average value in that interval,
                # which might help to improve the S/N ratio
                    #print(t_tmp, t_flag + dt/2.0, sum(por_tmp)/len(por_tmp), por_tmp)
                #elif len(por_tmp) == 1:
                    #por_edit.append(por_tmp)
                    #print(t_flag)
                else:
                    por_edit.append((por[nsp] + por[nsp-1])/2.0)
                    #print(t_flag)

                #print(t_flag)

                t_flag += dt
                por_tmp = []
                #print(len(por_edit))

            if t_tmp > t_max_tmp + dt/2.0:
                break

            nsp += 1
        #print(len(por_edit))
        por_final.append(por_edit)
        idx += 1

    return por_final

# Return any general logs that needs to be converted to time domain
def td_convert_v1(data_in, depth, dt_table, dt, t_min, t_max):
    # depth-time conversion

    dpt = depth
    table = dt_table
    t_min_tmp = t_min - dt
    t_max_tmp = t_max
    coef = linfit_v1(table, dt, t_min_tmp, t_max_tmp)

    time_seis = dpt * coef[0] + coef[1]  # different wells have different TD table

    # print( 4500* coef[0] + coef[1])  # The maxixium time available.
    #print(time_seis)
    t_flag = t_min_tmp
    nsp = 0
    log_edit = []
    log_tmp = []
    for t_tmp in time_seis:
        # print(t_tmp)

        if t_tmp >= t_flag and t_tmp < t_flag + dt:
            log_tmp.append(data_in[nsp])
            # print(por[np])

        if t_tmp >= t_flag + dt:

            # print(len(por_tmp))
            # print(por_tmp)
            if len(log_tmp) > 0:
                #por_edit.append(sum(por_tmp) / len(por_tmp))  # Calculate the target log use the average value in that interval,
                log_edit.append(log_tmp[0])
                # which might help to improve the S/N ratio
            # print(t_tmp, t_flag + dt/2.0, sum(por_tmp)/len(por_tmp), por_tmp)
            # elif len(por_tmp) == 1:
            # por_edit.append(por_tmp)
                #print(t_tmp, t_flag, log_tmp[0])
            else:
                log_edit.append((data_in[nsp] + data_in[nsp - 1]) / 2.0)
                #print(t_tmp, t_flag, (data_in[nsp] + data_in[nsp - 1]) / 2.0)

            t_flag += dt
            log_tmp = []
            # print(len(por_edit))

        if t_tmp > t_max_tmp + dt / 2.0:
            break
        nsp += 1
    data_edit = np.asarray(log_edit)
    return data_edit

# Convert MD to TVD using td table
def tvd_conv(data_in, dt_table, depth0, depth1):
    md = []
    tvd = []
    Nt = len(dt_table)
    for nt in range(Nt):
        if dt_table[nt, 0] >= depth0 and dt_table[nt, 0] <= depth1:
            md.append(dt_table[nt, 0])
            tvd.append(dt_table[nt, 1])

    md =  np.asarray(md)
    tvd = np.asarray(tvd)
    #print(md.shape, tvd.shape)
    fit_coef = np.polyfit(md, tvd, 1)
    data_out = data_in * fit_coef[0] + fit_coef[1]
    print(data_in, depth0, depth1, fit_coef, data_out)
    return data_out

def list_edit(MSFT_t):
    MSFT_V2 = []
    for MSFT_real in MSFT_t:
        MSFT_V1 = []
        for MSFT_tmp in MSFT_real:
            MSFT = []
            for tmp in MSFT_tmp:
                # print(tmp.shape)
                tmp = np.asarray(tmp)
                #print(tmp.shape)
                MSFT.append(tmp)
            MSFT_V1.append(MSFT)

        MSFT_V2.append(MSFT_V1)
    return MSFT_V2

def list_edit_v1(MSFT_ori):
    MSFT_V2 = []
    for MSFT_real in MSFT_ori:
        MSFT_V1 = []
        for MSFT_tmp in MSFT_real:
            MSFT_tmp = np.asarray(MSFT_tmp)
            MSFT_V1.append(MSFT_tmp)

        MSFT_V2.append(MSFT_V1)
    return MSFT_V2

def list_edit_v2(data_ori):
    data_V2 = []
    for data_tmp in data_ori:
        data_tmp = np.asarray(data_tmp)
        data_V2.append(data_tmp)
    return data_V2

def DataPrepV1(MSFT, dt, f_intv, Fout):
    fs = int(1 / dt * 1000)  # sampling frequency, Hz
    N_f = int(fs * 2)  # number of FFT Point, greater or equal to the sample number of hanning window?
    df = fs / N_f

    # calculate the output of MSFT for the corresponding frequency band
    fmax = int(Fout / df)
    fmin = int((Fout - f_intv) / df)
    MSFT_out = np.sum(MSFT[:, fmin:fmax, :, :], axis=1)  # 10 to 20 Hz for all phase.
    return MSFT_out

def freq_sep(por_final, MSFT_img_t, MSFT_real_t, Fmax, f_intv, dt):
    Nfq = int(Fmax / f_intv) + 1
    FOUT = np.linspace(f_intv, Fmax, num=Nfq)
    fs = int(1 / dt * 1000)  # sampling frequency, Hz

    MSFT_all = []
    # PORLF_all = []  # used to held frequency less than 10 Hz for porosity
    Pord_all = []  # list for narrow frequency band log

    for Fout in FOUT:
        MSFT = []
        PORLF = []  # used to held frequency less than 10 Hz for porosity
        Pord = []  # list for narrow frequency band log
        Ntr = len(MSFT_img_t)
        for nr in range(Ntr):
            MSFT_real = np.asarray(MSFT_real_t[nr])
            #print(MSFT_real.shape)
            MSFT_real = np.transpose(MSFT_real, (2, 3, 1, 0))
            MSFT_img = np.asarray(MSFT_img_t[nr])
            MSFT_img = np.transpose(MSFT_img, (2, 3, 1, 0))
            #print(MSFT_img.shape)

            MSFT_real_tmp = np.transpose(DataPrepV1(MSFT_real, dt, f_intv, Fout))
            MSFT_img_tmp = np.transpose(DataPrepV1(MSFT_img, dt, f_intv, Fout))
            MSFT_tmp = np.concatenate((MSFT_real_tmp, MSFT_img_tmp), axis=1)
            MSFT_tmp = np.transpose(MSFT_tmp, (2, 1, 0))
            #print(MSFT_tmp.shape)
            MSFT.append(MSFT_tmp)

            Por = np.asarray(por_final[nr])
            #print(Por.shape, Por)
            f1 = Fout - f_intv
            f2 = Fout
            if Fout < 1:
                # calculate the signal from 0 to 10 Hz
                b, a = signal.butter(3, Fout, btype="lp", fs=fs, output='ba')
                #print(Por)
                Por_10 = signal.filtfilt(b, a, Por)
                Pord.append(Por_10)
            else:
                b, a = signal.butter(3, [f1, f2], btype="bandpass", fs=fs, output='ba')
                por_temp = signal.filtfilt(b, a, Por)
                Pord.append(por_temp)

        MSFT_all.append(MSFT)
        Pord_all.append(Pord)

        # each well composite trace has diffenent length, so they won't be able to convert to numpy array
        # Pord = np.array(Pord)  # frequency band is only from 10 -100, no low frequency (0-10Hz)
        # PORLF = np.array(PORLF) # frequency: 0 - 10Hz
        # MSFT = np.array(MSFT)

        #print(MSFT.shape)
        Fout = int(Fout)
        if Fout < 100:
            FileName1 = ".\seismic attribute\MSFT_00" + str(Fout)
            FileName2 = ".\seismic attribute\POR_00" + str(Fout)
        else:
            FileName1 = ".\seismic attribute\MSFT_0" + str(Fout)
            FileName2 = ".\seismic attribute\POR_0" + str(Fout)

        #print(FileName1)
        #print(MSFT[0])
        np.save(FileName1, MSFT)
        np.save(FileName2, Pord)
        print("Fout =", Fout)
    return MSFT_all, Pord_all

def data_merge(MSFT, POR):
    Ntr = len(MSFT)
    MSFT_merge = MSFT[0]
    POR_merge = POR[0]
    for ntr in range(Ntr-1):
        MSFT_merge  = np.concatenate((MSFT_merge , MSFT[ntr+1]), axis=0)
        POR_merge = np.concatenate((POR_merge, POR[ntr+1]), axis=0)

    Ns, Nf, Nag = np.shape(MSFT_merge)
    MSFT_merge = np.reshape(MSFT_merge, (Ns, Nf*Nag))
    return MSFT_merge, POR_merge

# merge data for conventional attributes
def data_flat(data_in):
    Ntr = len(data_in)
    data_out = data_in[0]
    for ntr in range(Ntr-1):
        print(data_out.shape, data_in[ntr+1].shape)
        data_out  = np.concatenate((data_out, data_in[ntr+1]), axis=0)
    return data_out

def normalize(data_in):
    data_dim = np.shape(data_in)
    data_nor = []
    if len(data_dim) > 1:
        data_in = np.transpose(data_in)  # python and c is row base language, so make sure access the outer most data first. the second dimension (right most) changes fastest.
        for ns in range(data_in.shape[0]):
            mean_data = np.mean(data_in[ns, :])
            std_data = np.std(data_in[ns, :])
            data_tmp = (data_in[ns, :] - mean_data) / std_data
            data_nor.append(data_tmp)
        data_nor = np.asarray(data_nor)

    else:
        mean_data = np.mean(data_in)
        std_data = np.std(data_in)
        data_nor = (data_in - mean_data) / std_data
    return data_nor

def params_clc(data_in):
    data_dim = np.shape(data_in)
    data_nor = []
    std_all = []
    mean_all = []
    min_all = []
    max_all = []
    if len(data_dim) > 1:
        data_in = np.transpose(data_in)  # python and c is row base language, so make sure access the outer most data first. the second dimension (right most) changes fastest.
        for ns in range(data_in.shape[0]):
            mean_data = np.mean(data_in[ns, :])
            std_data = np.std(data_in[ns, :])
            min_tmp = min(data_in[ns, :])
            max_tmp = max(data_in[ns, :])
            std_all.append(std_data)
            mean_all.append(mean_data)
            min_all.append(min_tmp)
            max_all.append(max_tmp)
    par_all = [std_all, mean_all, min_all, max_all]
    par_all = np.asarray(par_all)
    return par_all

def unnormalize(data_in, mean, std):
    data_re = data_in * std + mean  # unnormalization
    return data_re

#min-max normalization
def normalize_v1(data_in):
    data_dim = np.shape(data_in)
    data_nor = []
    if len(data_dim) > 1:
        for ns in range(data_dim[1]):
            data_max = np.max(data_in[:, ns])
            data_min = np.min(data_in[:, ns])
            data_tmp = (data_in[:, ns] - data_min) / (data_max - data_min)
            data_nor.append(data_tmp)
        data_nor = np.asarray(data_nor)

    else:
        data_max = np.max(data_in)
        data_min = np.min(data_in)
        data_nor = (data_in - data_min) / (data_max - data_min)
    return data_nor

def unnormalize_v1(data_in, data_max, data_min):
    data_re = data_in * (data_max - data_min) + data_min  # unnormalization
    return data_re


# Write report to cvs file
def out_file(data_in, out_file_name, out_header):
    out_file = open(out_file_name, "w")
    M, N = np.shape(data_in)
    # Add header for output file
    out_file.write(out_header)

    # Write data into a cvs file
    for i in range(M):
        for j in range(N):
            data_tmp = str(data_in[i, j])
            if len(data_tmp) > 6:
                data_tmp = data_tmp[0:8]
            out_file.write(data_tmp)
            out_file.write('      ')
        out_file.write('\n')

    out_file.close()

def freq_iso(logs, Fout, dt):
    NW = len(logs)
    Nt = len(logs[0])
    fs = int(1 / dt * 1000)  # sampling frequency, Hz
    N_f = int(fs)  # number of FFT Point, greater or equal to the sample number of hanning window?
    df = fs / N_f
    NFout = int((Fout+1)/df)
    xt_r_all = []

    for nw in range(NW):
        log_tmp = logs[nw]
        Xw = np.fft.fft(log_tmp, n=N_f)  # N_f has to be 1/dt to get the correct ifft
        xt_r = []
        pi = 3.1415926
        for nt in range(Nt):
            ts = nt * dt  # need to be very careful!!
            xt_f = []
            for nw in range(NFout):
                f = nw * df
                f1 = fs - f
                xt_tmp = (np.real(Xw[nw]) * np.cos(2 * pi * f * ts * 0.001) - np.imag(Xw[nw]) * np.sin(2 * pi * f * ts * 0.001))
                if nw == 0.0:
                    xt_tmp1 = 0.0
                else:
                    xt_tmp1 = (np.real(Xw[nw]) * np.cos(2 * pi * f1 * ts * 0.001) + np.imag(Xw[nw]) * np.sin(2 * pi * f1 * ts * 0.001))  # the imaginary part is odd function, so need to change sign here.

                #if nw > -1:
                xt_tmp0 = (xt_tmp + xt_tmp1)/fs
                xt_f.append(xt_tmp0)
            xt_r.append(xt_f)
        xt_r_all.append(xt_r)
    xt_r_all = np.asarray(xt_r_all)
    return xt_r_all

def freq_iso_sgl(logs, Fout, dt):
    Nt = len(logs)
    fs = int(1 / dt * 1000)  # sampling frequency, Hz
    N_f = int(fs)  # number of FFT Point, greater or equal to the sample number of hanning window?
    df = fs / N_f
    NFout = int((Fout+1)/df)
    Xw = np.fft.fft(logs, n=N_f)  # N_f has to be 1/dt to get the correct ifft
    xt_r = []
    pi = 3.1415926
    for nt in range(Nt):
        ts = nt * dt  # need to be very careful!!
        xt_f = []
        for nw in range(NFout):
            f = nw * df
            f1 = fs - f
            xt_tmp = (np.real(Xw[nw]) * np.cos(2 * pi * f * ts * 0.001) - np.imag(Xw[nw]) * np.sin(
                2 * pi * f * ts * 0.001))
            if nw == 0.0:
                xt_tmp1 = 0.0
            else:
                xt_tmp1 = (np.real(Xw[nw]) * np.cos(2 * pi * f1 * ts * 0.001) + np.imag(Xw[nw]) *
                           np.sin(
                               2 * pi * f1 * ts * 0.001))  # the imaginary part is odd function, so need to change sign here.

            # if nw > -1:
            xt_tmp0 = (xt_tmp + xt_tmp1) / fs
            xt_f.append(xt_tmp0)
        xt_r.append(xt_f)
    xt_r_all = np.asarray(xt_r)
    return xt_r_all



