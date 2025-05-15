import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """Takes Number And Always Provide Same Vector of size d_mdoel. In paper d_mdoel is 512
    This vector is learned by the model - but which model?
    """
    def __init__(self,d_model: int ,vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size =  vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
    
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model) # this is divided by the root d as decivbed in the paper


class PositionalEncoding(nn.Module):
    """ Think. This is the postion for each vector. So its shape should match the vector of the Input Embeddings"""
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        """ Squence length is the maximum length of the sentence"""
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        #create a matrix of lemgth (seq_len,d_model) -> why ? i need a vecot of same size as 512 but how many ? of tthe sequence length.
        pe = torch.zeros(seq_len,d_model)

        #create  a vector of shape (seq_len)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) #create a tensor of seq_len,1
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))

        #apply sine in even dim and cosine in the odd term 
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        #adding batch dimernsioon batch,sequence,d_model
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        "in x.shape first is batch , secodn is the sequence length and 3rd d_model vector"
        x = x + (self.pe[:,:x.shape[1], :]).requires_grad(False) #so that tensor do not learn the positional embedding its fixed
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #multilied
        self.bias = nn.Parameter(torch.zeros(1)) #added
    
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim= True) #clacaute the mean of the the dimension that is not the batch  but keep the dimension
        std = x.std(dim=1-1,keepdim=True)
        return self.alpha * (x-mean)/(std+self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int,dropout:float):
        super().__init__()
        self.liner_1 = nn.Linear(d_model,d_ff)
        self.droput = nn.Dropout(dropout)
        self.liner_2 = nn.Linear(d_ff,d_model)


    def forward(self,x):
        #Batch,seq_len,dmodel --> linear1 ---> batch,seq_len,d_ff --->  linear2 ---> batch,seq_len,d_model
        return self.liner_2(self.droput(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float):
        """ h is number of head """
        """ each head will have the full access to the sentence but only will have access to part of their embeddings"""
        """ so the understnd that division happens in the embedding dim  no the seq dimension"""
        """ to make sure the the each head gets equal size matrix d_model should be divisible by the head"""
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # this coresposding to what each head sees
        self.w_q = nn.Linear(d_model,d_model) #wq
        self.w_k = nn.Linear(d_model,d_model) #wk
        self.w_v = nn.Linear(d_model,d_model) #wv

        self.w_o  = nn.Linear(d_model,d_model) #in paper the W0 metric is (h*dv, d_model)  #dv and dk are of same size dv is used for wehn dk is multiplied by v matrix. Also yeh h*dv is d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self,q,k,v,mask):
        """mask is used when we dont want some words to interact with other. Most of the time duiring inference
        basicially we calacute the attention score for each q intretatcing with each v. so mask is used to make the number small so that during the atteanion softamax calcates it as 0
        """
        query = self.w_q(q) #q_prime -> (batch,seq_len,d_model)
        key = self.w_k(k) #k_prime -> (batch,seq_len,d_model)
        value = self.w_v(v) #v_prime -> (batch,seq_len,d_model)

        #(Batch,seq_len,d_model) -> botach,seq_Len,h,d_k   --> (batch,h,Seq_len,d_k)  
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(query.value[0],query.value[1],self.h,self.d_k).transpose(1,2)

        
          









    








