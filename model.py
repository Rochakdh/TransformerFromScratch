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
    
    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1]
        attention_score = (query @ key.transpose(-2,-1))/math.sqrt(d_k) #mat mul of query (batch,h,Seq_len,d_k) * (batch,h,d_k,seq_len) => (batch,h,seq_len(from query),seq_len(from value))
        if mask is not None:
            """mask is needed when we dontr what token to attend future token like in decoder
            or if we dont want padding value to interact with other value they are just filler words to reach sequence length
            """
            attention_score.mask_fill_(mask == 0, -1e9) #mask is defined in such a way that where mask ==0  will be replaced by specified value
        attention_score = attention_score.softmax(dim=-1) #(Batch,h,seq_len(from query),seq_len(from key)) # sommax is only apply to the last dimension.
        #This applies softmax across the last dimension, which is L_key — i.e., across all key positions for each query.
        """For each query vector, softmax across keys tells the model:Given this query, how much attention should I pay to each position in the sequence?"""
        """Softmax is applied per row, where each row corresponds to how a specific token attends to all others."""
        if dropout is not None:
            attention_score = dropout(attention_score)
        return (attention_score @ value),attention_score



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

        x,self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        # (B,h,seq_len,d_k) ---- for concat bring back to ---> (B,seq_len,h,d_k) ------> (B,seq_Len,d_model)
        x = x.transpose(1,2).contigious().view(x.shape[0],-1,self.h * self.d_k)


        return self.w_o(x)


class ResidualConnection(nn.Nodules):
    def __init__(self, dropout : float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection  = nn.ModuleList([ResidualConnection(dropout)for _ in range(2)])

    def forward(self,x,src_mask):
        x = self.residual_connection[0](x,lambda x: self.self_attention_block(x,x,x,src_mask)) #in encoder same senetant is use as query key value each watching each
        x = self.residual_connection[1](x,self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask) 
        return self.norm(x)



class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.Module([ResidualConnection(dropout)for _ in range(3)])
    
    def forward(self,x,encoder_output,src_mask,tgt_mask): #src is applied to encode and tgt is applied to decoder we have soource lange ie. english and a targte language
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))#apply target mask from the decoder
        x = self.residual_connection[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask)) # query from decoder key and valeu from encoder,mask from encoder
        x = self.residual_connection[2](x,self.feed_forward_block)



class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layer = layers
        self.norm = LayerNormalization()

    def forward(self,x,encoder_output,src_msk,tgt_msk):
        for layer in self.layers:
            x = layer(x,encoder_output,src_msk,tgt_msk)
        return self.norm(x)


    


    


    


    


          









    








