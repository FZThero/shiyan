import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as Data
from torch import nn
import function as fc
class ICInn(nn.Module): #网络模型 ，其中注意力机制中间层数是个超参数，后面dnn网络激活函数是个超参数，算法学习率基本是0.001/超参。层数和隐藏层神经元个数超参
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
        # for i in range(14):  # 取一个子帧14个符号
        atten_out = self.attention(x)
        c = torch.mul(atten_out, x)
        out = self.zuihou(c)
        return out

class BGRUnn(nn.Module):
    def __init__(self, feature, hidden_size, num_layers): #(104,128,1)
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

    def forward(self, x, h_0): #x:(128,14,104) h0:(2,128,128)
        out1 = self.attention(x)
        y, h_n = self.rnn(out1, h_0)  #x:(128,14,104) ，hn(2,128,128)
        # out1 = torch.zeros(size=(200, 14, 144)).cuda()
        # for i in range(14):  # 取一个子帧14个符号
        #    out1[:,i,:] = self.attention(y[:, i, :]).cuda()
        # out1 = self.attention(y)
        # out2,h_n2 = self.rnn2(out1,h_0)
        batch, seq_L, hidden_size2 = y.shape #(128,14,144)
        y = y.contiguous() #和view是一对的
        y = y.view(-1, hidden_size2)#（1792，256）
        y2 = self.fc(y)
        y2 = y2.view(batch, seq_L, self.out_size) #（128，14，104）
        return y2

all_carrier = 72
frame_length = 14
multipath = 7

# 设定一些数据的数目和标志位
test_idx_low = 1
test_idx_high =2001

test_num = 2000
# save_flag = 1
load_flag = 1
train_flag = 0
test_flag = 1
train_SNRdb = 20
test_SNRdb = [0, 5, 10, 15, 20, 25]
pilot_carrier_index = [0, 4, 9]
all_carrier_index = np.arange(14)
data_carrier_index = np.delete(all_carrier_index, pilot_carrier_index)
BATCH_SIZE = 50
gen_test = 1

# PATH = 'Attention_BGRU.pth.tar'
Load_Path = 'Attention_BGRU.pth.tar'
MAX_EPOCH = 100
LR = 0.001
num_layers = 1
hidden_size = 100
feature = 144
test_MSE = np.zeros(len(test_SNRdb), dtype=float)
# 设定模型 损失函数 优化算法

model = BGRUnn(feature, hidden_size, num_layers).cuda()
print(model)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
# 用来加载训练好的模型  /   设定容限误差
if load_flag:
    print("Loading Model")
    checkpoint = torch.load(Load_Path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # model.eval() #这个就是用在测试中，防止调用一些语句导致一些错误的发生
else:
    epoch = 0
    loss_min = 100000


if gen_test:
    for index in range(len(test_SNRdb)):
        # data = np.zeros((test_num, frame_length, all_carrier * multipath), dtype=complex)# （500，14，72*5）
        channel_response_set_test, channel_response_set_test_label, channel_response_set_test_redata = \
            fc.getting_channel_response4(test_idx_low, test_idx_high, gen_test,test_SNRdb[index])
        for i in range(2000):
            data = channel_response_set_test[i]
            receive_data = channel_response_set_test_redata[i]

            channel_mse = fc.suanfa(model,data,data_carrier_index,receive_data)#(1008,5)
            channel_mse2 = np.concatenate((np.real(channel_mse), np.imag(channel_mse)), axis=1)
            np.savetxt('Attention_BGRU_matric150kmh/channelSNR'+str(test_SNRdb[index])+'dB'+str(i+1)+'.txt',channel_mse2)
        print('test_data saved')
