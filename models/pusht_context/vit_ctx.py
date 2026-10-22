"""Noncausal Context attention, preserving the bound learned-position layout."""
import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self,hidden_dim,mlp_hidden_dim,dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,mlp_hidden_dim),
                                 nn.GELU(),nn.Dropout(dropout),
                                 nn.Linear(mlp_hidden_dim,hidden_dim),nn.Dropout(dropout))

    def forward(self,x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,hidden_dim,n_heads=8,head_dim=64,dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.scale = head_dim**-0.5
        self.norm = nn.LayerNorm(hidden_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(hidden_dim,3*n_heads*head_dim,bias=False)
        self.to_out = nn.Identity() if n_heads == 1 and head_dim == hidden_dim else nn.Sequential(
            nn.Linear(n_heads*head_dim,hidden_dim),nn.Dropout(dropout))

    def forward(self,x,rope_cos=None,rope_sin=None,n_cls=0):
        batch,length = x.shape[:2]
        q,k,v = [value.reshape(batch,length,self.n_heads,-1).transpose(1,2)
                 for value in self.to_qkv(self.norm(x)).chunk(3,-1)]
        probability = self.dropout(self.attend((q @ k.transpose(-1,-2))*self.scale))
        mixed = (probability @ v).transpose(1,2).reshape(batch,length,-1)
        return self.to_out(mixed)


class Transformer(nn.Module):
    def __init__(self,hidden_dim,depth,n_heads,head_dim,mlp_hidden_dim,dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([
            Attention(hidden_dim,n_heads,head_dim,dropout),
            FeedForward(hidden_dim,mlp_hidden_dim,dropout)]) for _ in range(depth)])

    def forward(self,x,rope_cos=None,rope_sin=None,n_cls=0):
        for attention,feed_forward in self.layers:
            x = x + attention(x)
            x = x + feed_forward(x)
        return x


class ViT_Ctx(nn.Module):
    def __init__(self,*,in_dim,out_dim,hidden_dim,depth,n_heads,mlp_hidden_dim,
                 head_dim=64,max_seq_len=1024,use_rope=False,dropout=0.0,emb_dropout=0.0):
        super().__init__()
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        self.to_token_embedding = nn.Sequential(nn.LayerNorm(in_dim),nn.Linear(in_dim,hidden_dim),
                                                nn.LayerNorm(hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(hidden_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1,max_seq_len+1,hidden_dim))
        self.transformer = Transformer(hidden_dim,depth,n_heads,head_dim,mlp_hidden_dim,dropout)
        self.mlp_head = nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,out_dim))
        self.output_dim = out_dim

    def forward(self,tokens):
        x = self.to_token_embedding(tokens)
        batch,length = x.shape[:2]
        x = torch.cat((self.cls_token.expand(batch,1,-1),x),dim=1)
        x = self.dropout(x+self.pos_embedding[:,:length+1])
        return self.mlp_head(self.transformer(x)[:,0])
