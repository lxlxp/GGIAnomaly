import torch
import torch.nn as nn
class GRUEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(GRUEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.GRU(input_size=c_in, hidden_size=512)




    def forward(self, x):

        x,hidden= self.tokenConv(x)

        return x