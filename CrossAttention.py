import string
import numpy as np
import pandas as pd 
from random import randint
import math
## PyTorch
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

class CrossAttention:
  def __init__(self,input_1,input_2):
    self.input_1 = input_1
    self.input_2 = input_2


  def text_embedding(self):
    dc_1 = {s:i for i,s in enumerate(sorted(self.input_1.replace(',', '').split()))}
    dc_2 = {s:i for i,s in enumerate(sorted(self.input_2.replace(',', '').split()))}

    sentence_1 = torch.tensor([dc_1[s] for s in self.input_1.replace(',', '').split()])
    sentence_2 = torch.tensor([dc_2[s] for s in self.input_2.replace(',', '').split()])

    emebedding_1 = torch.nn.Embedding(4,10)
    emebedding_2 = torch.nn.Embedding(2,10)

    X_1 = emebedding_1(sentence_1).detach()
    X_2 = emebedding_2(sentence_2).detach()
    return X_1, X_2

  def weight_initiliazation(self):
    d = self.text_embedding()[0].shape[1]

    W_q = torch.nn.Parameter(torch.rand(d, d))
    W_k = torch.nn.Parameter(torch.rand(d, d))
    W_v = torch.nn.Parameter(torch.rand(d, d))

    return W_q, W_k, W_v

  def get_Q_K_and_V(self,W_q, W_k, W_v ):
    Q = X_1 @ W_q
    K = X_2 @ W_k
    V = X_2 @ W_v
    return Q, K, V

  def scaled_dot_product(self,Q, K, V, mask=None):
    d = Q.shape[1]
    attn_logits = torch.matmul(Q, K.T)
    attn_logits = attn_logits / math.sqrt(d)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, V)
    return values, attention

  def fit(self):
    X_1,X_2 = self.text_embedding()
    W_q, W_k, W_v = self.weight_initiliazation()
    Q, K, V = self.get_Q_K_and_V(W_q, W_k, W_v )
    values, attention  = self.scaled_dot_product(Q, K, V, mask=None)
    print(" Outputs \n\n",values)
    print("Attention Matrix \n\n",attention)
