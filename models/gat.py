import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class GraphAttentionLayer(nn.Module):
    """

    图注意力层
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True,batchsize=32):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征维度
        self.out_features = out_features  # 节点表示向量的输出特征维度
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活


        self.W = nn.Parameter(torch.zeros(size=(batchsize,in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化
        self.batchsize=batchsize
        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):

        h = torch.matmul(inp, self.W)  # [N, out_features]
        N = h.size()[1]  # N 图的节点数

        a_input = torch.cat([h.repeat(1,1, N).view(self.batchsize,N * N, -1), h.repeat(1,N, 1)], dim=1).view(self.batchsize,N, -1, 2 * self.out_features)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))


        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')' \
                                                                                                          '' \
                                                                                                          '' \
                                                                                                    ''
x=torch.randn(32,38,96)

x1 = np.ones((32,38, 38))
adj=torch.tensor(x1)
GAL=GraphAttentionLayer(96,96,0.2,0.2)
print(GAL(x,adj).shape)
