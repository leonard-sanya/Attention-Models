import string
import numpy as np
import pandas as pd 
from random import randint
import math
## PyTorch
import torch # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

class MultiheadAttention:
  def __init__(self,heads,data):
    self.heads =heads
    self.data = data

  def text_embedding(self):
    dc = {s:i for i,s in enumerate(sorted(self.data.replace(',', '').split()))}
    sentence_int = torch.tensor([dc[s] for s in self.data.replace(',', '').split()])
    emebedding = torch.nn.Embedding(len(sentence_int),128)
    X = emebedding(sentence_int).detach()
    return X
  def weight_initiliazation(self):
    d = self.text_embedding().shape[1]

    W_q = []
    W_k = []
    W_v = []

    for i in range(self.heads):
      Wi_q = torch.nn.Parameter(torch.rand(d, int(d/self.heads)))
      Wi_k = torch.nn.Parameter(torch.rand(d, int(d/self.heads)))
      Wi_v = torch.nn.Parameter(torch.rand(d, int(d/self.heads)))

      W_q.append(Wi_q)
      W_k.append(Wi_k)
      W_v.append(Wi_v)
    return W_q, W_k, W_v

  def get_Q_K_and_V(self,W_q, W_k, W_v ):

    Q = []
    K = []
    V = []

    for i in range(self.heads):
      q = X @ W_q[i]
      k = X @ W_k[i]
      v = X @ W_v[i]

      Q.append(q)
      K.append(k)
      V.append(v)
    return Q, K, V

  def Scaled_dot_product(self,Q, K, V, mask=None):
    d = Q.shape[1]
    attn_logits = torch.matmul(Q, K.T)
    attn_logits = attn_logits / math.sqrt(d)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, V)
    return values, attention

  def fit(self):

    X = self.text_embedding()
    W_q, W_k, W_v = self.weight_initiliazation()
    Q, K, V = self.get_Q_K_and_V(W_q, W_k, W_v )

    Heads = []
    for i in range(self.heads):
      values, attention = self.Scaled_dot_product(Q[i], K[i], V[i])
      Heads.append(values)
    output = Heads[0].detach()
    for i in range(1,self.heads):
      output = np.concatenate((output, Heads[i].detach()), axis=1)

    print(" Outputs",output)
    
