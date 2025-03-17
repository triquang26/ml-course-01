import torch
from torch import nn
from torch import nn, einsum
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
import torchvision
import time
from torchinfo import summary
from torchvision import datasets, transforms
import torch
from torch import nn
from torch import nn, einsum
import torch.nn.functional as F
from torch import optim

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import torchvision
import time
from torchinfo import summary
from medmnist import PneumoniaMNIST

import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def scaled_dot_product(V,Q,K,Fk,mask = None) :
  C = torch.matmul(Q,K.transpose(-2,-1))/ torch.sqrt(torch.tensor(Fk, dtype=torch.float32))
  if mask is not None:
      scores = scores.masked_fill(mask == 0, float('-1e9'))
  C = nn.Softmax(dim=-1)(C)
  return torch.matmul(C,V)

class NewAttention(nn.Module):
    def __init__(self, Fv, Fk, Fv_o, Fk_o, heads,F_out, drop_out = 0.1,final_linear = True):
        super(NewAttention, self).__init__()
        self.Fv = Fv
        self.Fk = Fk
        self.Fv_o = Fv_o
        self.Fk_o = Fk_o
        self.heads = heads
        self.head_linearV = nn.Linear(self.Fv,self.Fv_o* self.heads)
        self.head_linearK = nn.Linear(self.Fk,self.Fk_o* self.heads)
        self.head_linearQ = nn.Linear(self.Fk,self.Fk_o* self.heads)
        self.final_linear = nn.Linear(heads * Fv_o, F_out)
        self.droput = nn.Dropout(drop_out)
    def forward(self, V, K, Q, mask=None):
        # V, K, Q: [batch, seq_len, dim]
        batch, seq_len, _ = V.shape
        v = self.head_linearV(V)
        k = self.head_linearK(K)
        q = self.head_linearQ(Q) ### [N,S,F_v] -> [N,S,Fv_o*heads]

        v = v.reshape(batch,seq_len, self.heads,self.Fv_o).transpose(1, 2) #[N,S,Fv_o*heads]  -> [N,S,heads,Fv_o] -> [N,heads,S,Fv_o]
        k = k.reshape(batch,seq_len, self.heads,self.Fk_o).transpose(1, 2)
        q = q.reshape(batch, seq_len, self.heads, self.Fk_o).transpose(1, 2)

        attn = scaled_dot_product(v, q, k, self.Fk_o,mask) # [batch, heads, seq_len, Fv_o]
        attn = attn.transpose(1,2).reshape(batch,seq_len, self.heads * self.Fv_o) # [batch, seq_len, Fv_o * heads]
        result = self.final_linear(attn.reshape(batch,seq_len, self.heads * self.Fv_o))
        return result


class FF (nn.Module):
    def __init__(self, F_in,F_o,dropout_rate = 0.1):
        super(FF, self).__init__()
        self.linear1 = nn.Linear(F_in,F_o)
        self.linear2 = nn.Linear(F_o,F_in)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class Transformer(nn.Module):
    def __init__(self, F_in,F_o,mlp_dim,heads,dropout_rate = 0.1):  ### check up  on this
      # self.
      super(Transformer, self).__init__()
      self.attn_norm = nn.LayerNorm(F_in, eps=1e-6)
      self.norm = nn.LayerNorm(F_in, eps=1e-6)
      self.ff = FF(F_in,mlp_dim,dropout_rate)
      self.attention = NewAttention(F_in,F_in,F_o,F_o,heads,F_in)
      # self.attention = NewAttention(F_in,heads)
    def forward(self,x,mask = None):
      residue = x
      x = self.attn_norm(x)
      x = self.attention(x,x,x)
      # x = self.attention(x)
      x = x + residue
      residue = x
      x = self.norm(x)
      x = self.ff(x)
      x = x + residue
      return x

class VIT(nn.Module) :
  def __init__(self,channel,image_h, image_w, patch, F_in,F_o,F_out, heads,trans_depth,mlp_dim, mode = "average"):
    super(VIT, self).__init__()
    assert image_h % patch == 0, "Image height must be divisible by the patch size."
    assert image_w % patch == 0, "Image width must be divisible by the patch size."
    self.channel = channel
    self.image_h = image_h
    self.image_w = image_w
    self.patch = patch
    self.num_patch_h = image_h // patch
    self.num_patch_w = image_w // patch
    self.num_patch = self.num_patch_h * self.num_patch_w
    self.mode = mode

    patch_dim = channel * patch * patch
    self.patch_linear = nn.Linear(patch_dim, F_in)
    self.trans_layers = nn.ModuleList([Transformer(F_in,F_o,mlp_dim,heads) for i in range(trans_depth)])

    self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patch + 1, F_in))

    self.cls_token = nn.Parameter(torch.randn(1, 1, F_in))
    self.dropout = nn.Dropout(0.1)
    self.norm = nn.LayerNorm(F_in, eps=1e-6)
    self.final_linear  = nn.Linear(F_in,F_out)

  def forward(self,x,mask = None):
    b, c, h, w = x.shape
    x = x.reshape(b, c, self.num_patch_h, self.patch, self.num_patch_w, self.patch)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [b, num_patch_h, num_patch_w, c, patch, patch]
    x = x.reshape(b, self.num_patch, c * self.patch * self.patch)  # [b, num_patch, patch_dim]
    x = self.patch_linear(x)
    cls_tokens = self.cls_token.repeat(b, 1, 1)
    x = torch.cat([cls_tokens, x], dim=1)
    x += self.pos_embedding
    x = self.dropout(x)
    for layer in self.trans_layers:
      x = layer(x)
    x = self.norm(x)
    x = self.final_linear(x)
    if self.mode == "average":
      return x.mean(dim = 1)
    else : return x[:,0,:]
  def save_model(self, filepath):
    """
    Save the model's state dictionary to the given filepath.
    
    Parameters:
      filepath (str): The path to the file where the state dictionary will be saved.
    """
    torch.save(self.state_dict(), filepath)
  @classmethod
  def load_model(cls, filepath):
      """
      Load the entire model from a .pth file.
      
      Parameters:
          filepath (str): The file path to the saved model.
      
      Returns:
          VIT: The loaded model instance.
      """
      model = torch.load(filepath, map_location=torch.device('cpu'))
      model.eval()  # Optionally set to evaluation mode after loading
      return model