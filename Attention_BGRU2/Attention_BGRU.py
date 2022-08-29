import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as Data
from matplotlib import pyplot as plt
from torch import nn
import function as fc
import time
class ICInn(nn.Module):
    def __init__(self):
        super(ICInn, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(144, 9),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(9, 144),
            nn.Sigmoid(),)
        self.zuihou = nn.Sequential(
            nn.Linear(144,144),
        )


    def forward(self, x):
        # out = torch.zeros(size=(200, 14, 144)).cuda()
        # for i in range(14):
        atten_out = self.attention(x)
        c = torch.mul(atten_out, x)
        out = self.zuihou(c)
        return out

class BGRUnn(nn.Module):
    def __init__(self, feature, hidden_size, num_layers):
        super(BGRUnn, self).__init__()
        self.feature = feature
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = 144
        self.attention = ICInn()
        self.rnn = nn.GRU(
            input_size=feature,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size*2,self.out_size),
        )

    def forward(self, x, h_0):
        out1 = self.attention(x)
        y, h_n = self.rnn(out1, h_0)
        # out1 = torch.zeros(size=(200, 14, 144)).cuda()
        # for i in range(14):  # 取一个子帧14个符号
        #    out1[:,i,:] = self.attention(y[:, i, :]).cuda()
        # out1 = self.attention(y)
        # out2,h_n2 = self.rnn2(out1,h_0)
        batch, seq_L, hidden_size2 = y.shape
        y = y.contiguous()
        y = y.view(-1, hidden_size2)
        y2 = self.fc(y)
        y2 = y2.view(batch, seq_L, self.out_size)
        return y2



def train_model(model, train_loader,train_num,valid_loader, valid_num, MAX_EPOCH, last_epoch, loss_func, loss_min, optimizer):
    print('train begin')
    train_loss = np.zeros(MAX_EPOCH)#(100)
    mse_record = np.zeros(MAX_EPOCH)#(100)
    for epoch in range(last_epoch, MAX_EPOCH):#(0-100)
        for step, (x,y) in enumerate(train_loader):
            b_x = x.requires_grad_().cuda()
            b_y = y.cuda()
            batch, K, xL = b_x.size()
            _, K2, yL = b_y.size()
            optimizer.zero_grad()

            h_0 = torch.zeros(2, batch, 100).cuda()
            # c_0 = torch.zeros(1, batch,144).cuda()
            prediction = torch.zeros(size=(batch,K2,yL)).cuda()


            prediction[:,:,:] = model(b_x,h_0)


            loss = loss_func(prediction, b_y)
            train_loss[epoch] += loss
            loss.backward()
            optimizer.step()


        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), },
                   PATH + '/m-' + str(epoch + 1) + '.pth.tar')
        print('Epoch: ', epoch + 1)

        mse_record[epoch] = valid_model(model, valid_loader, valid_num)


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(np.arange(0, MAX_EPOCH) + 1, train_loss, 'k')
    ax1.set_ylabel('Train Loss')
    ax1.set_title("Train Loss & MSE")
    ax1.set_xlabel("Epoch")
    ax2 = ax1.twinx() #共享x轴
    ax2.plot(np.arange(0, MAX_EPOCH) + 1, mse_record, 'r-o')
    ax2.set_ylabel('MSE')
    ax2.set_xlabel('Epoch')
    plt.savefig('result_test/MSE_curve/train_and_valid_Attention_BGRU', dpi=600)
    plt.show()
    return 0
#
def valid_model(model, valid_loader, valid_num):
    mse = 0
    print('test begin')
    for step, (x, y) in enumerate(valid_loader):
        b_x = x.requires_grad_().cuda()
        b_y = y.cuda()
        batch, K, xL = b_x.size()
        _, K2, yL = b_y.size()

        h_0 = torch.zeros(2, batch, 100).cuda()
        # c_0 = torch.zeros(1, batch, 144).cuda()
        prediction = torch.zeros(size=(batch, K2, yL)).cuda()

        prediction[:, :, :] = model(b_x,h_0)


        pred = prediction.cpu()
        pred = pred.detach().numpy()
        y = y.detach().numpy()
        pred2 = pred[:, :, :72] + 1j * pred[:, :, 72:]
        y2 = y[:, :, :72] + 1j * y[:, :, 72:]
        mse += np.sum(np.abs(pred2[:, :, :] - y2[:, :, :]) ** 2 / (valid_num * 72 * 14))

    print('MSE:', mse)
    return mse
#
def test_model(model,test_loader,test_num):
    mse = 0
    print('test begin')
    for step, (x, y) in enumerate(test_loader):
        b_x = x.requires_grad_().cuda()
        b_y = y.cuda()
        batch, K, xL = b_x.size()
        _, K2, yL = b_y.size()

        h_0 = torch.zeros(2, batch, 100).cuda()
        # c_0 = torch.zeros(1, batch, 144).cuda()
        prediction = torch.zeros(size=(batch, K2, yL)).cuda()

        prediction[:, :, :] = model(b_x,h_0)



        pred = prediction.cpu()
        pred = pred.detach().numpy()
        y = y.detach().numpy()
        pred2 = pred[:,:,:72]+1j*pred[:,:,72:]
        y2 = y[:,:,:72]+1j*y[:,:,72:]
        mse += np.sum(np.abs(pred2[:, :,:] - y2[:, :,:]) ** 2 /(test_num * 72 * 14))
        # pred2 = pred[:,:,:72*5]+1j*pred[:,:,72*5:]
        # y2 = y[:,:,:72*5]+1j*y[:,:,72*5:]
    print('MSE:', mse)
    return mse

if __name__ == "__main__":
    all_carrier = 72
    frame_length = 14
    multipath = 7

    #设定一些数据的数目和标志位
    train_num =15000#12000
    train_num_half = 7500#6000
    valid_num =3000
    valid_num_half = 1500
    test_num = 2000
    load_flag = 1
    train_flag = 0
    test_flag = 1
    train_SNRdb = 20
    test_SNRdb = [0,5,10,15,20,25]
    pilot_carrier_index = [0,4,9]
    all_carrier_index = np.arange(14)
    data_carrier_index = np.delete(all_carrier_index, pilot_carrier_index)
    pilot_path_index1 = np.array([0,1,2,3,4,5,6])
    pilot_path_index2 = np.array([28,29,30,31,32,33,34])
    pilot_path_index3 = np.array([63,64,65,66,67,68,69])
    pilot_path_index = np.concatenate([pilot_path_index1,pilot_path_index2,pilot_path_index3],axis=0)
    BATCH_SIZE = 50

    # PATH = 'Attention_BGRU/Attention_BGRU.pth.tar'
    Load_Path = 'Attention_BGRU.pth.tar'
    MAX_EPOCH = 100
    LR = 0.001
    test_MSE = np.zeros(len(test_SNRdb), dtype=float)
    #设定模型 损失函数 优化算法
    num_layers = 1
    hidden_size = 100
    feature = 144
    # model1 = ICInn().cuda()
    model = BGRUnn(feature, hidden_size, num_layers).cuda()
    print(model)
    #
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay = 1e-6)
    #用来加载训练好的模型  /   设定容限误差
    if load_flag:
        print("Loading Model")
        checkpoint = torch.load(Load_Path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = 0
        # model.eval() #在测试中，防止调用一些语句导致一些错误的发生
    else:
        epoch = 0
        loss_min = 100000
    #
    if train_flag:
        train_label = np.load("train_label72carrier/train_label_" + str(train_SNRdb) + "db.npy")
        train_data = np.load("train_data72carrier/train_data_" + str(train_SNRdb) + "db.npy")


        valid_label = np.load("train_label72carrier/valid_label_" + str(train_SNRdb) + "db.npy")
        valid_data = np.load("train_data72carrier/valid_data_" + str(train_SNRdb) + "db.npy")

        print('Train & Validation Data Loaded')
        train_data_final = np.concatenate((np.real(train_data),np.imag(train_data)),axis=2)#（12000，14，72*2）

        train_label_final = np.concatenate((np.real(train_label), np.imag(train_label)), axis=2)
        # train_label_final = train_label_final.reshape(train_num,28,72)

        valid_data_final = np.concatenate((np.real(valid_data), np.imag(valid_data)),axis=2)  # （12000，14，72*2）

        valid_label_final = np.concatenate((np.real(valid_label), np.imag(valid_label)), axis=2)
        # valid_label_final = valid_label_final.reshape(valid_num, 28, 72)

        # valid_data_final = np.concatenate((np.real(valid_data), np.imag(valid_data),np.real(valid_re_data),np.imag(valid_re_data)),axis=2)#（2000，98，72*2）
        #
        # valid_label_final = np.concatenate((np.real(valid_label), np.imag(valid_label)), axis=2)

        # train_data_final = train_data_final.reshape(train_num,14*4,72)
        # valid_data_final = valid_data_final.reshape(valid_num,14*4,72)

        train_label_final = torch.from_numpy(train_label_final).float()  # 取浮点型换为张量格式
        train_data_final = torch.from_numpy(train_data_final).float()
        valid_label_final = torch.from_numpy(valid_label_final).float()
        valid_data_final = torch.from_numpy(valid_data_final).float()

        dataset_train = Data.TensorDataset(train_data_final, train_label_final)
        train_loader = Data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        dataset_valid = Data.TensorDataset(valid_data_final, valid_label_final)
        valid_loader = Data.DataLoader(dataset=dataset_valid, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        train_model(model, train_loader, train_num, valid_loader, valid_num, MAX_EPOCH,epoch, loss_func, loss_min, optimizer)
    #
    if test_flag:
        for index in range(len(test_SNRdb)):
            test_label = np.load("test_label_150kmh/test_label_" + str(test_SNRdb[index]) + "db.npy")
            test_data = np.load("test_data_150kmh/test_data_" + str(test_SNRdb[index]) + "db.npy")
            # test_re_data = np.load("test_data_new200receive_data_" + str(test_SNRdb[index]) + "db.npy")
            # test_T_data = np.load("test_data_new200/T_data_" + str(test_SNRdb[index]) + "db.npy")
            print('Test Data Loaded')

            test_data_final = np.concatenate((np.real(test_data), np.imag(test_data)), axis=2)  # （12000，14，72*2）
            test_label_final = np.concatenate((np.real(test_label), np.imag(test_label)), axis=2)

            # test_data_final = test_data_final.reshape(test_num, 14 * 4, 72)

            test_label_final = torch.from_numpy(test_label_final).float()  # 取浮点型换为张量格式
            test_data_final = torch.from_numpy(test_data_final).float()

            dataset_test = Data.TensorDataset(test_data_final, test_label_final)
            test_loader = Data.DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

            test_MSE[index] = test_model(model, test_loader, test_num)
        np.savetxt('result_test/MSE_value/Attention_BGRU.txt', test_MSE)


        plt.xlabel('SNR_db')
        plt.ylabel('Channel Estimation MSE')
        plt.title('Performance of Channel Estimator')
        plt.xscale('linear')
        plt.yscale('log')
        plt.plot(test_SNRdb, test_MSE, color="blue", marker='s', linestyle="-", linewidth=1, label="Attention_BGRU")
        plt.legend(loc='upper right')
        plt.grid(b=True, which='both')
        plt.savefig('result_test/MSE_curve/MSE_Attention_BGRU_without', dpi=600)
        plt.show()
