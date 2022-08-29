import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as Data
from torch import nn
import function as fc


all_carrier = 72
frame_length = 14
multipath = 7
train_idx_low = 1
train_idx_high = 15001
valid_idx_low = 1
valid_idx_high = 3001
test_idx_low = 1
test_idx_high = 2001
#设定一些数据的数目和标志位
train_num =15000#12000
train_num_half = 7500#6000
valid_num =3000
valid_num_half = 1500
test_num =2000
save_flag = 1
gen_train = 0
gen_test =1
train_SNRdb = 20
test_SNRdb = [0,5,10,15,20,25]
pilot_carrier_index = [0,4,9]
all_carrier_index = np.arange(14)
data_carrier_index = np.delete(all_carrier_index, pilot_carrier_index)
snr = 20
print('Getting Channel Response')
channel_response_set_train,channel_response_set_train_label= \
    fc.getting_channel_response(train_idx_low, train_idx_high,gen_train,snr) #（12000）list，每个list（140，64）
print('Train Channel Response Generated')#训练数组： 10000个每一个（1008，5）

channel_response_set_valid,channel_response_set_valid_label= \
    fc.getting_channel_response3(valid_idx_low, valid_idx_high,gen_train,snr) #（12000）list，每个list（140，64）
print('Valid Channel Response Gen'
      'erated')#训练数组： 10000个每一个（1008，5）


if gen_train:

    label = np.zeros((train_num, 14,72),dtype=complex)  # （12000，14，72*5）
    data = np.zeros((train_num, 14,72), dtype=complex)
    # receive_data = np.zeros((train_num, 14, 72), dtype=complex)
    valid_label = np.zeros((valid_num, 14,72),dtype=complex)  # （12000，14，72*5）
    valid_data = np.zeros((valid_num, 14,72), dtype=complex)
    # valid_receive_data = np.zeros((valid_num, 14, 72), dtype=complex)

    # train_data_float = np.zeros((train_num, 14*72, 2), dtype=float)
    # train_label_float = np.zeros((train_num, 14*72, 2), dtype=float)
    # valid_data_float = np.zeros((valid_num, 14 * 72, 2), dtype=float)
    # valid_label_float = np.zeros((valid_num, 14 * 72, 2), dtype=float)
    for i in range(15000):
        # data[i,:,:] = channel_response_set_train[i].reshape(14,all_carrier*multipath)
        # label[i, :, :] = channel_response_set_train_label[i].reshape(14, all_carrier * multipath)
        data[i,:,:] = channel_response_set_train[i]
        label[i, :, :] = channel_response_set_train_label[i]
        # receive_data[i, :, :] = channel_response_set_train_redata[i]
    for k in range(3000):
        valid_data[k, :, :] = channel_response_set_valid[k]
        valid_label[k, :, :] = channel_response_set_valid_label[k]
        # valid_receive_data[k, :, :] = channel_response_set_valid_redata[k]
    # train_label_float[:, :, 0] = np.real(label.reshape(train_num,-1))
    # train_label_float[:, :, 1] = np.imag(label.reshape(train_num,-1))
    # train_data_float[:, :, 0] = np.real(data.reshape(train_num,-1))
    # train_data_float[:, :, 1] = np.imag(data.reshape(train_num,-1))
    # train_data_float[:, :, 2] = np.imag(data.reshape(train_num,-1))
    # train_data_float[:, :, 3] = np.imag(receive_data.reshape(train_num,-1))

    # valid_label_float[:, :, 0] = np.real(valid_label.reshape(valid_num,-1))
    # valid_label_float[:, :, 1] = np.imag(valid_label.reshape(valid_num, -1))
    # valid_data_float[:, :, 0] = np.real(valid_data.reshape(valid_num, -1))
    # valid_data_float[:, :, 1] = np.imag(valid_data.reshape(valid_num, -1))
    # valid_data_float[:, :, 2] = np.imag(valid_data.reshape(valid_num, -1))
    # valid_data_float[:, :, 3] = np.imag(valid_receive_data.reshape(valid_num, -1))
    if save_flag:
        # np.save("train_data3.6/train_data_" + str(train_SNRdb) + "db.npy", data[:train_num,:,:])
        # np.save("train_label3.6/train_label_" + str(train_SNRdb) + "db.npy", label[:train_num,:,:])  # 注意在for循环下，每个信噪比生成一个文件
        # np.save("train_data3.6/valid_data_" + str(train_SNRdb) + "db.npy", data[train_num:,:,:])
        # np.save("train_label3.6/valid_label_" + str(train_SNRdb) + "db.npy",label[train_num:,:,:])
        # np.save("train_data3.6/receive_data_" + str(train_SNRdb) + "db.npy", receive_data[:train_num,:,:])
        # np.save("train_data3.6/receive_data_valid" + str(train_SNRdb) + "db.npy", receive_data[train_num:,:,:])
        np.save("train_data_72carrier/train_data_" + str(train_SNRdb) + "db.npy", data)
        np.save("train_label_72carrier/train_label_" + str(train_SNRdb) + "db.npy",label)  # 注意在for循环下，每个信噪比生成一个文件
        np.save("train_data_72carrier/valid_data_" + str(train_SNRdb) + "db.npy", valid_data)
        np.save("train_label_72carrier/valid_label_" + str(train_SNRdb) + "db.npy",valid_label)
        # np.save("train_data_new4/receive_data_" + str(train_SNRdb) + "db.npy", receive_data)
        # np.save("train_data_new4/receive_data_valid" + str(train_SNRdb) + "db.npy", valid_receive_data)
    print('train_data saved')

if gen_test:
    for index in range(len(test_SNRdb)):
        # label = np.zeros((test_num, frame_length, all_carrier * multipath),dtype=complex)  # （600，14，72*5）
        # data = np.zeros((test_num, frame_length, all_carrier * multipath), dtype=complex)
        label = np.zeros((test_num, 14, 72), dtype=complex)  # （600，14，72*5）
        data = np.zeros((test_num, 14,72), dtype=complex)
        snr = test_SNRdb[index]
        channel_response_set_test, channel_response_set_test_label =\
            fc.getting_channel_response2(test_idx_low, test_idx_high, gen_test,snr)
        # test_data_float = np.zeros((test_num, 14 * 72, 2), dtype=float)
        # test_label_float = np.zeros((test_num, 14 * 72, 2), dtype=float)
        for i in range(2000):
            # data[i,:,:] = channel_response_set_test[i].reshape(14,all_carrier*multipath)
            # label[i, :, :] = channel_response_set_test_label[i].reshape(14, all_carrier * multipath)
            data[i, :, :] = channel_response_set_test[i]
            label[i, :, :] = channel_response_set_test_label[i]
        # test_label_float[:, :, 0] = np.real(label.reshape(test_num, -1))
        # test_label_float[:, :, 1] = np.imag(label.reshape(test_num, -1))
        # test_data_float[:, :, 0] = np.real(data.reshape(test_num, -1))
        # test_data_float[:, :, 1] = np.imag(data.reshape(test_num, -1))
        if save_flag:
            np.save("test_data_150kmh/test_data_" + str(test_SNRdb[index]) + "db.npy", data)
            np.save("test_label_150kmh/test_label_" + str(test_SNRdb[index]) + "db.npy", label)
        print('test_data saved')