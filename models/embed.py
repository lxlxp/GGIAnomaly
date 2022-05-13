import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import tensorflow as tf
from keras_layer_normalization import LayerNormalization
from tensorflow import assert_rank
# tf.enable_eager_execution(
#     config=None,
#     device_policy=None,
#     execution_mode=None
# )
from models.gat import GraphAttentionLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape
def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)

def print_shape(tensor, rank, tensor_name):
  return tensor
  tensor_shape = get_shape_list(tensor, expected_rank=rank)
  return tf.Print(tensor, [tensor_shape], tensor_name, summarize=8)




class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
#位置编码32*96*512
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class GRUEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(GRUEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.GRU(input_size=c_in, hidden_size=d_model)




    def forward(self, x):

        x,hidden= self.tokenConv(x)

        return x

class GATEmbedding(nn.Module):
    def __init__(self, c_in, d_model,lenth):
        super(GATEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.ts=c_in
        self.ts2 = lenth
        encoder_layer = nn.TransformerEncoderLayer(d_model=lenth, nhead=2)
        self.tokenConv=nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.GATConv = GraphAttentionLayer(lenth,lenth,0.2,0.2).cuda()
        self.x1 = np.ones((32, 38, 38))
        self.adj = torch.tensor(self.x1)
        self.adj = self.adj.to("cuda:0")



    def forward(self, x):
        #print(x.shape)


        feature_gat_input = torch.transpose(x,1,2)

        feature_gat_output1 = self.GATConv(feature_gat_input,self.adj)

        feature_gat_output = torch.transpose(feature_gat_output1, 1, 2)
        #print(feature_gat_output.shape)
        return feature_gat_output


#经过一个1维卷积，输入从32*96*7到32*96*512
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x
#32*96*4到32*96*512
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1,len=96):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=2*c_in, d_model=d_model)
        #self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        #self.gru_embedding=GRUEmbedding(c_in=2*c_in, d_model=d_model)
        self.gru_embedding = GRUEmbedding(c_in=c_in, d_model=d_model)
        self.gat_embedding = GATEmbedding(c_in=c_in, d_model=d_model,lenth=len)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):


        x0=self.gat_embedding(x)

        C=torch.cat([x,x0],2)

        x = self.value_embedding(C) + self.position_embedding(C)+self.temporal_embedding(x_mark)

        return self.dropout(x)
    #相加