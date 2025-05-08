import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """Takes Number And Always Provide Same Vector of size d_mdoel. In paper d_mdoel is 512\
    This vector is learned by the model - but which model?
    """
    def __init__(self,d_model: int ,vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size =  vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
    
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model) 


