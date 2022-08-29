#用来生成导频值
import copy
import math

import numpy as np
import torch
from scipy.interpolate import interpolate
import numpy.linalg as lg


def Pilot_Modulation(original_data,modulation):
    if modulation=='QAM':
        original_data = original_data.reshape((int(len(original_data) / 2), 2))
    return (2 * original_data[:, 0] - 1) + 1j * (2 * original_data[:, 1] - 1)  # This is just for QAM modulation


#得到信道响应
def getting_channel_response(train_idx_low, train_idx_high,gen_train,snr):
    channel_response_set_train = []
    channel_response_set_train_label = []
    # channel_response_set_train_redata=[]
    if gen_train:
        H_folder = 'H_train_datanew72carrier/channelSNR'
        for train_idx in range(train_idx_low, train_idx_high):#
            H_file = H_folder + str(snr) + 'dB' + str(train_idx) + '.txt'
            # H_file = H_folder + str(train_idx) + '.txt'
            with open(H_file) as f:
                h_response = np.zeros((14,72),dtype=complex) #6TAP
                count = 0
                for line in f: #一次取一行
                    if count < 14:
                        numbers_str = line.split()  #将每一行的两列数据分开
                        numbers_float = [float(x) for x in numbers_str] #每个数取浮点类型
                        h_response[count,:] = np.asarray(numbers_float[0:int(len(numbers_float) / 2)]) + 1j * np.asarray(numbers_float[int(len(numbers_float) / 2):len(numbers_float)])#一行合成一个复数，asarray比array占用内存小
                        count = count + 1
                channel_response_set_train.append(h_response) #一共12000个

        H_folder2 = 'H_train_labelnew72carrier/channelSNR'
        for train_idx2 in range(train_idx_low, train_idx_high):#
            H_file2 = H_folder2 + str(snr) + 'dB' + str(train_idx2) + '.txt'
            # H_file2 = H_folder2 + str(train_idx2) + '.txt'
            with open(H_file2) as f:
                h_response2 = np.zeros((14,72),dtype=complex) #6TAP
                count = 0
                for line in f: #一次取一行
                    if count < 14:
                        numbers_str = line.split()  #将每一行的两列数据分开
                        numbers_float = [float(x) for x in numbers_str] #每个数取浮点类型
                        h_response2[count,:] = np.asarray(numbers_float[0:int(len(numbers_float) / 2)]) + 1j * np.asarray(numbers_float[int(len(numbers_float) / 2):len(numbers_float)])#一行合成一个复数，asarray比array占用内存小
                        count = count + 1
                channel_response_set_train_label.append(h_response2) #一共12000个

        # H_folder3 = 'R_test_data4/channelSNR'
        # for train_idx3 in range(train_idx_low, train_idx_high):  #
        #     H_file3 = H_folder3 + str(snr) + 'dB' + str(train_idx3) + '.txt'
        #     with open(H_file3) as f:
        #         h_response3 = np.zeros((14, 72), dtype=complex)  # 6TAP
        #         count = 0
        #         for line in f:  # 一次取一行
        #             if count < 14:
        #                 numbers_str = line.split()  # 将每一行的两列数据分开
        #                 numbers_float = [float(x) for x in numbers_str]  # 每个数取浮点类型
        #                 h_response3[count, :] = np.asarray(
        #                     numbers_float[0:int(len(numbers_float) / 2)]) + 1j * np.asarray(numbers_float[int(len(
        #                     numbers_float) / 2):len(numbers_float)])  # 一行合成一个复数，asarray比array占用内存小
        #                 count = count + 1
        #         channel_response_set_train_redata.append(h_response3)  # 一共12000个

    return channel_response_set_train,channel_response_set_train_label

def getting_channel_response3(valid_idx_low, valid_idx_high,gen_train,snr):
    channel_response_set_valid = []
    channel_response_set_valid_label = []
    # channel_response_set_valid_redata=[]
    if gen_train:
        H_folder = 'H_valid_datanew72carrier/channelSNR'
        for valid_idx in range(valid_idx_low, valid_idx_high):#
            H_file = H_folder + str(snr) + 'dB' + str(valid_idx) + '.txt'
            # H_file = H_folder + str(valid_idx) + '.txt'
            with open(H_file) as f:
                h_response = np.zeros((14,72),dtype=complex) #6TAP
                count = 0
                for line in f: #一次取一行
                    if count < 14:
                        numbers_str = line.split()  #将每一行的两列数据分开
                        numbers_float = [float(x) for x in numbers_str] #每个数取浮点类型
                        h_response[count,:] = np.asarray(numbers_float[0:int(len(numbers_float) / 2)]) + 1j * np.asarray(numbers_float[int(len(numbers_float) / 2):len(numbers_float)])#一行合成一个复数，asarray比array占用内存小
                        count = count + 1
                channel_response_set_valid.append(h_response) #一共12000个

        H_folder2 = 'H_valid_labelnew72carrier/channelSNR'
        for valid_idx2 in range(valid_idx_low, valid_idx_high):#
            H_file2 = H_folder2 + str(snr) + 'dB' + str(valid_idx2) + '.txt'
            # H_file2 = H_folder2 + str(valid_idx2) + '.txt'
            with open(H_file2) as f:
                h_response2 = np.zeros((14,72),dtype=complex) #6TAP
                count = 0
                for line in f: #一次取一行
                    if count < 14:
                        numbers_str = line.split()  #将每一行的两列数据分开
                        numbers_float = [float(x) for x in numbers_str] #每个数取浮点类型
                        h_response2[count,:] = np.asarray(numbers_float[0:int(len(numbers_float) / 2)]) + 1j * np.asarray(numbers_float[int(len(numbers_float) / 2):len(numbers_float)])#一行合成一个复数，asarray比array占用内存小
                        count = count + 1
                channel_response_set_valid_label.append(h_response2) #一共12000个

        # H_folder3 = 'R_valid_data4/channelSNR'
        # for valid_idx3 in range(valid_idx_low, valid_idx_high):#
        #     H_file3 = H_folder3 + str(snr) + 'dB' + str(valid_idx3) + '.txt'
        #     with open(H_file3) as f:
        #         h_response3 = np.zeros((14,72),dtype=complex) #6TAP
        #         count = 0
        #         for line in f: #一次取一行
        #             if count < 14:
        #                 numbers_str = line.split()  #将每一行的两列数据分开
        #                 numbers_float = [float(x) for x in numbers_str] #每个数取浮点类型
        #                 h_response3[count,:] = np.asarray(numbers_float[0:int(len(numbers_float) / 2)]) + 1j * np.asarray(numbers_float[int(len(numbers_float) / 2):len(numbers_float)])#一行合成一个复数，asarray比array占用内存小
        #                 count = count + 1
        #         channel_response_set_valid_redata.append(h_response3) #一共12000个

    return channel_response_set_valid,channel_response_set_valid_label


def getting_channel_response2(test_idx_low, test_idx_high, gen_test,snr):
    channel_response_set_test = []
    channel_response_set_test_label = []
    if gen_test:
        H_folder = 'H_test_datanew72carrier150kmh/channelSNR'
        for test_idx in range(test_idx_low, test_idx_high):  #
            H_file = H_folder + str(snr) +'dB'+ str(test_idx) + '.txt'
            with open(H_file) as f:
                h_response = np.zeros((14, 72), dtype=complex)  # 6TAP
                count = 0
                for line in f:  # 一次取一行
                    if count < 14:
                        numbers_str = line.split()  # 将每一行的两列数据分开
                        numbers_float = [float(x) for x in numbers_str]  # 每个数取浮点类型
                        h_response[count, :] = np.asarray(
                            numbers_float[0:int(len(numbers_float) / 2)]) + 1j * np.asarray(numbers_float[int(len(
                            numbers_float) / 2):len(numbers_float)])  # 一行合成一个复数，asarray比array占用内存小
                        count = count + 1
                channel_response_set_test.append(h_response)  # 一共12000个

        H_folder2 = 'H_test_labelnew72carrier150kmh/channelSNR'
        for test_idx2 in range(test_idx_low, test_idx_high):  #
            H_file2 = H_folder2 + str(snr) +'dB'+ str(test_idx2) + '.txt'
            with open(H_file2) as f:
                h_response2 = np.zeros((14, 72), dtype=complex)  # 6TAP
                count = 0
                for line in f:  # 一次取一行
                    if count < 14:
                        numbers_str = line.split()  # 将每一行的两列数据分开
                        numbers_float = [float(x) for x in numbers_str]  # 每个数取浮点类型
                        h_response2[count, :] = np.asarray(
                            numbers_float[0:int(len(numbers_float) / 2)]) + 1j * np.asarray(numbers_float[int(len(
                            numbers_float) / 2):len(numbers_float)])  # 一行合成一个复数，asarray比array占用内存小
                        count = count + 1
                channel_response_set_test_label.append(h_response2)  # 一共12000个


    return channel_response_set_test, channel_response_set_test_label


def getting_channel_response4(test_idx_low, test_idx_high, gen_test,snr):
    channel_response_set_test = []
    channel_response_set_test_label = []
    channel_response_set_test_redata = []

    if gen_test:
        H_folder = 'H_test_datanew72carrier150kmh/channelSNR'
        for test_idx in range(test_idx_low, test_idx_high):#
            H_file = H_folder + str(snr) +'dB'+ str(test_idx) + '.txt'
            with open(H_file) as f:
                h_response = np.zeros((14,72),dtype=complex) #6TAP
                count = 0
                for line in f:
                    if count < 14:
                        numbers_str = line.split()  #将每一行的两列数据分开
                        numbers_float = [float(x) for x in numbers_str] #每个数取浮点类型
                        h_response[count,:] = np.asarray(numbers_float[0:int(len(numbers_float) / 2)]) + 1j * np.asarray(numbers_float[int(len(numbers_float) / 2):len(numbers_float)])#一行合成一个复数，asarray比array占用内存小
                        count = count + 1
                channel_response_set_test.append(h_response) #一共12000个

        H_folder2 = 'H_test_labelnew72carrier150kmh/channelSNR'
        for test_idx2 in range(test_idx_low, test_idx_high):#
            H_file2 = H_folder2 + str(snr) +'dB'+ str(test_idx2) + '.txt'
            with open(H_file2) as f:
                h_response2 = np.zeros((14,72),dtype=complex) #6TAP
                count = 0
                for line in f: #一次取一行
                    if count < 14:
                        numbers_str = line.split()  #将每一行的两列数据分开
                        numbers_float = [float(x) for x in numbers_str] #每个数取浮点类型
                        h_response2[count,:] = np.asarray(numbers_float[0:int(len(numbers_float) / 2)]) + 1j * np.asarray(numbers_float[int(len(numbers_float) / 2):len(numbers_float)])#一行合成一个复数，asarray比array占用内存小
                        count = count + 1
                channel_response_set_test_label.append(h_response2) #一共12000个

        H_folder3 = 'R_test_data_new72carrier150kmh/channelSNR'
        for test_idx3 in range(test_idx_low, test_idx_high):  #
            H_file3 = H_folder3 + str(snr) +'dB'+ str(test_idx3) + '.txt'
            with open(H_file3) as f:
                h_response3 = np.zeros((14, 72), dtype=complex)  # 6TAP
                count = 0
                for line in f:  # 一次取一行
                    if count < 14:
                        numbers_str = line.split()  # 将每一行的两列数据分开
                        numbers_float = [float(x) for x in numbers_str]  # 每个数取浮点类型
                        h_response3[count, :] = np.asarray(
                            numbers_float[0:int(len(numbers_float) / 2)]) + 1j * np.asarray(numbers_float[int(len(
                            numbers_float) / 2):len(numbers_float)])  # 一行合成一个复数，asarray比array占用内存小
                        count = count + 1
                channel_response_set_test_redata.append(h_response3)  # 一共12000个

    return channel_response_set_test, channel_response_set_test_label, channel_response_set_test_redata
#产生随机信号的
def generate_signal(m, n):
    output = np.random.randint(0, 2, size=[m, n])    #np.random.randint（a,b） 的取值范围为[a，b）之间的整数
    return output

#采用调制方式的设置
def map(signal_bit, modu_way):
    '''
    :param signal_bit: the bit signal ,shape = (ofdm_sym_num, data_sub_num*bit_to_sym[modu_way])
    :param modu_way:  0:bpsk, 1:qpsk, 2:16qam, 3:64qam
    :return: output , pilot_symbol
             output = signal_symbol, shape =(ofdm_sym_num, data_sub_num)
    '''

    if modu_way == 0:
        output = map_bpsk(signal_bit)
    elif modu_way == 1:
        output = map_qpsk(signal_bit)
    elif modu_way == 2:
        output = map_16qam(signal_bit)
    elif modu_way == 3:
        output = map_64qam(signal_bit)
    else:
        print('the input of modu_way is error')
        output = 1
    return output


def map_bpsk(signal_bit):
    output = np.empty_like(signal_bit, dtype="complex64")
    for m in range(signal_bit.shape[0]):
        for n in range(signal_bit.shape[1]):
            if signal_bit[m, n] == 0:
                output[m, n] = -1 + 0j
            else:
                output[m, n] = 1 + 0j
    return output


def map_qpsk(signal_bit):
    c = int(signal_bit.shape[0])
    d = int(signal_bit.shape[1] / 2)
    x = signal_bit.reshape(c, d, 2)
    output = np.empty((c, d), dtype="complex64")
    for m in range(c):
        for n in range(d):
            a = x[m, n, :]
            if (a == [0, 0]).all():
                output[m, n] = complex(-math.sqrt(2)/2, -math.sqrt(2)/2)
            elif (a == [0, 1]).all():
                output[m, n] = complex(-math.sqrt(2)/2, math.sqrt(2)/2)
            elif (a == [1, 1]).all():
                output[m, n] = complex(math.sqrt(2) / 2, math.sqrt(2) / 2)
            else:
                output[m, n] = complex(math.sqrt(2) / 2, -math.sqrt(2) / 2)
    return output


def map_16qam(signal_bit): #需要看？？？？？？？？？？？？？
    c = int(signal_bit.shape[0])
    d = int(signal_bit.shape[1]/4)
    x = signal_bit.reshape(c, d, 4)
    output = np.empty((c, d), dtype="complex64")  #得到一个对应维度的空数组
    for m in range(c):#每次取一个符号
        for n in range(d):#每次取每个符号的子载波数值
            a = x[m, n, :2]  #取的是小于二的位置，也就是得到前两个值
            if (a == [0, 0]).all():  #all用法是看里面元素是否都为TRUE，若是返回TRUE
                real = -3   #？？？？？？
            elif (a == [0, 1]).all():
                real = -1
            elif (a == [1, 1]).all():
                real = 1
            else:
                real = 3
            b = x[m, n, 2:]
            if (b == [0, 0]).all():
                imag = -3
            elif (b == [0, 1]).all():
                imag = -1
            elif (b == [1, 1]).all():
                imag = 1
            else:
                imag = 3
            output[m, n] = complex(real, imag)/math.sqrt(10)
    return output


def map_64qam(signal_bit):
    c = int(signal_bit.shape[0])
    d = int(signal_bit.shape[1]/6)
    x = signal_bit.reshape(c, d, 6)
    output = np.empty((c, d), dtype="complex64")
    for m in range(c):
        for n in range(d):
            a = x[m, n, :3]
            if (a == [0, 0, 0]).all():
                real = -7
            elif (a == [0, 0, 1]).all():
                real = -5
            elif (a == [0, 1, 1]).all():
                real = -3
            elif (a == [0, 1, 0]).all():
                real = -1
            elif (a == [1, 0, 0]).all():
                real = 7
            elif (a == [1, 0, 1]).all():
                real = 5
            elif (a == [1, 1, 1]).all():
                real = 3
            else:
                real = 1
            b = x[m, n, 3:]
            if (b == [0, 0, 0]).all():
                imag = -7
            elif (b == [0, 0, 1]).all():
                imag = -5
            elif (b == [0, 1, 1]).all():
                imag = -3
            elif (b == [0, 1, 0]).all():
                imag = -1
            elif (b == [1, 0, 0]).all():
                imag = 7
            elif (b == [1, 0, 1]).all():
                imag = 5
            elif (b == [1, 1, 1]).all():
                imag = 3
            else:
                imag = 1
            output[m, n] = complex(real, imag)/math.sqrt(84)
    return output


def demap(signal_symbol, modu_way):
    '''
    :param signal_symbol: the symbol signal ,shape = (ofdm_sym_num, data_sub_num)
    :param modu_way:  0:bpsk, 1:qpsk, 2:16qam, 3:64qam
    :return: output
             output = signal_bit, shape =(ofdm_sym_num, data_sub_num*bit_to_sym[modu_way])
    '''
    if signal_symbol.ndim == 1:
        signal_symbol = signal_symbol[np.newaxis, :]
    if modu_way == 0:
        output = demap_bpsk(signal_symbol)
    elif modu_way == 1:
        output = demap_qpsk(signal_symbol)
    elif modu_way == 2:
        output = demap_16qam(signal_symbol)
    elif modu_way == 3:
        output = demap_64qam(signal_symbol)
    else:
        print('the input of modu_way is error')
        output = 1
    return output


def demap_bpsk(x):
    output = np.empty_like(x, dtype="int")
    for m in range(x.shape[0]):
        for n in range(x.shape[1]):
            if x[m, n].real >= 0:
                output[m, n] = 1
            else:
                output[m, n] = 0
    return output


def demap_qpsk(x):
    c = int(x.shape[0])
    d = int(x.shape[1])
    output = np.empty((c, d, 2), dtype="int")
    for m in range(c):
        for n in range(d):
            a = x[m, n].real
            b = x[m, n].imag
            if (a <= 0) & (b <= 0):
                output[m, n, :] = [0, 0]
            elif (a <= 0) & (b > 0):
                output[m, n, :] = [0, 1]
            elif (a > 0) & (b > 0):
                output[m, n, :] = [1, 1]
            else:
                output[m, n, :] = [1, 0]
    output = output.reshape(c, int(2*d))
    return output


def demap_16qam(x):
    c = int(x.shape[0])
    d = int(x.shape[1])
    output = np.empty((c, d, 4), dtype="int")
    for m in range(c):
        for n in range(d):
            a = math.sqrt(10)*x[m, n].real
            if a <= -2:
                output[m, n, :2] = [0, 0]
            elif (a <= 0) & (a > -2):
                output[m, n, :2] = [0, 1]
            elif (a <= 2) & (a > 0):
                output[m, n, :2] = [1, 1]
            else:
                output[m, n, :2] = [1, 0]
            b = math.sqrt(10)*x[m, n].imag
            if b <= -2:
                output[m, n, 2:] = [0, 0]
            elif (b <= 0) & (b > -2):
                output[m, n, 2:] = [0, 1]
            elif (b <= 2) & (b > 0):
                output[m, n, 2:] = [1, 1]
            else:
                output[m, n, 2:] = [1, 0]
    output = output.reshape((c, int(4*d)))
    return output


def demap_64qam(x):
    c = int(x.shape[0])
    d = int(x.shape[1])
    output = np.empty((c, d, 6), dtype="int")
    for m in range(c):
        for n in range(d):
            a = math.sqrt(84)*x[m, n].real
            if a <= -6:
                output[m, n, :3] = [0, 0, 0]
            elif (a > -6) & (a <= -4):
                output[m, n, :3] = [0, 0, 1]
            elif (a > -4) & (a <= -2):
                output[m, n, :3] = [0, 1, 1]
            elif (a > -2) & (a <= 0):
                output[m, n, :3] = [0, 1, 0]
            elif (a > 0) & (a <= 2):
                output[m, n, :3] = [1, 1, 0]
            elif (a > 2) & (a <= 4):
                output[m, n, :3] = [1, 1, 1]
            elif (a > 4) & (a <= 6):
                output[m, n, :3] = [1, 0, 1]
            else:
                output[m, n, :3] = [1, 0, 0]
            b = math.sqrt(84) * x[m, n].imag
            if b <= -6:
                output[m, n, 3:] = [0, 0, 0]
            elif (b > -6) & (b <= -4):
                output[m, n, 3:] = [0, 0, 1]
            elif (b > -4) & (b <= -2):
                output[m, n, 3:] = [0, 1, 1]
            elif (b > -2) & (b <= 0):
                output[m, n, 3:] = [0, 1, 0]
            elif (b > 0) & (b <= 2):
                output[m, n, 3:] = [1, 1, 0]
            elif (b > 2) & (b <= 4):
                output[m, n, 3:] = [1, 1, 1]
            elif (b > 4) & (b <= 6):
                output[m, n, 3:] = [1, 0, 1]
            else:
                output[m, n, 3:] = [1, 0, 0]
    output = output.reshape(c, int(6*d))
    return output

def cal_ber(x, y):
    ber = 0
    x1 = x.reshape(-1)  #变成一维向量(9216000)
    y1 = y.reshape(-1)#(9216000)
    error_num = 0
    for index in range(len(x1)):
        if x1[index] != y1[index]:
            error_num = error_num + 1
    ber = error_num
    return ber



#插入循环前缀
def insert_cp(ofdm_modulation_out, cp_length):
    output = np.concatenate((ofdm_modulation_out[:, :, -cp_length:], ofdm_modulation_out), axis=2)#从时频面来看是从下到上为一个一个符号
    return output

#信号经过信道时的卷积运算：
def add_channel(input, ht):
    output = np.zeros((140,85), dtype="complex64")#140 and 85 can be changed
    for m in range(input.shape[0]):
        output[m,:] = np.convolve(input[m,:],ht[m,:])

    return output

# def add_channel(input,ht):
# #加噪声
def wgn(x,snr):
    signal_power = np.mean(abs(x ** 2))
    sigma2 = signal_power * (10 ** (-snr / 10))
    noise = np.sqrt(sigma2/2) * (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape))
    return x + noise





def Insert(H_Estimate, train_num, pilot_carrier_index2,total_carrier,OFDM_Data_block,OFDM_block_length):  # H_Estimate
    H_Estimate2 = np.zeros((train_num,total_carrier*OFDM_block_length*OFDM_Data_block),dtype='complex64')
    H_Estimate3 = np.zeros((train_num,total_carrier*OFDM_Data_block*OFDM_block_length,2),dtype='float')
    for i in range(H_Estimate.shape[0]):#取总共的数量
        idx = np.arange(0, 52)
        pilot_carrier_index3 = copy.copy(pilot_carrier_index2)  # 防止更改原来的数组
        for j in range(14):
            x = pilot_carrier_index3
            y = H_Estimate[i,pilot_carrier_index3]
            f = interpolate.interp1d(x, y, kind='slinear',bounds_error=False,fill_value="extrapolate") #axis默认y的最后一个轴
            H_Estimate2[i,idx] = f(idx)
            pilot_carrier_index3 += 52
            idx += 52
    H_Estimate3[:,:,0] = np.real(H_Estimate2)
    H_Estimate3[:,:,1] = np.imag(H_Estimate2)
    return H_Estimate3 # 最后返回了一个完整的信道估计值，把0处给补齐了。



def Clipping (x,CL):
    out = np.zeros((140,80),dtype=complex)
    for i in range(140):
        sigma = np.sqrt(np.mean(np.square(np.abs(x[i,:]))))
        CL = CL*sigma
        x_clipped = x[i,:]
        t = abs(x_clipped)
        clipped_idx = np.where(t > CL)
        x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx]*CL),abs(x_clipped[clipped_idx]))
        out[i,:] = x_clipped
    return out

def suanfa(model,data,data_carrier_index,receive_data):
    test_data_final = np.concatenate((np.real(data), np.imag(data)), axis=1)
    # receive_data_final = np.concatenate((np.real(receive_data),np.imag(receive_data)),axis=1)

    test_data_final = torch.from_numpy(test_data_final).float()
    # receive_data_final = torch.from_numpy(receive_data_final).float()

    b_x = test_data_final.cuda()
    # b_z = receive_data_final.cuda()

    b_x = b_x.unsqueeze(0)
    # b_z = b_z.unsqueeze(0)
    h_0 = torch.zeros(2, 1, 100).cuda()
    prediction = torch.zeros(size=(1,14, 72*2)).cuda()  # （200，98，72*2）

    prediction[:, :, :] = model(b_x, h_0)

    pred = prediction.cpu()
    pred = pred.detach().numpy()
    pred2 = pred[:, :, :72] + 1j * pred[:, :, 72:]
    # pred3 = receive_data/pred2
    pred4 = pred2.squeeze(0)
    return pred4