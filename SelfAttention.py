import string
import numpy as np
import pandas as pd 
from random import randint
import math
## PyTorch
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore


class SelfAttention:
  def __init__(self,data1,mask = None):
    self.data1 = data1
    self.mask = mask


  def text_embedding(self):
    dc = {s:i for i,s in enumerate(sorted(self.data1.replace(',', '').split()))}
    sentence_int = torch.tensor([dc[s] for s in self.data1.replace(',', '').split()])
    emebedding = torch.nn.Embedding(len(sentence_int),128)
    X = emebedding(sentence_int).detach()
    return X


  def weight_initiliazation(self):
    d = self.text_embedding().shape[1]
    W_q = torch.nn.Parameter(torch.rand(d, d))
    W_k = torch.nn.Parameter(torch.rand(d, d))
    W_v = torch.nn.Parameter(torch.rand(d, d))
    return W_q, W_k, W_v

  def get_Q_K_and_V(self,X,W_q, W_k, W_v ):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

  def scaled_dot_product(self,Q, K, V):
    d = Q.shape[1]
    attn_logits = torch.matmul(Q, K.T)
    attn_logits = attn_logits / math.sqrt(d)
    if self.mask is not False:
      mask_simple = torch.tril(torch.ones(Q.shape[0], Q.shape[0]))
      attn_logits = attn_logits * mask_simple
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, V)
    print("Output",values)
    print("Atttention matrix",attention)

  def fit(self):
    X = self.text_embedding()
    W_q, W_k, W_v = self.weight_initiliazation()
    Q, K, V = self.get_Q_K_and_V(X,W_q, W_k, W_v )
    return self.scaled_dot_product(Q, K, V)



