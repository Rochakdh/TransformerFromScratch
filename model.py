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
        position = torch.arange(0,seq_len-1,dtype=torch.float).unsqueeze(1) #create a tensor of seq_len,1
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












