import math
import torch
import torch.nn as nn
import torch.nn.functional as F
            
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, emsize, nhead, nhid, nlayers, dropout=0.0):
        super(TransformerModel, self).__init__()

        self.ntoken = ntoken
        self.emsize = emsize
        self.nhead = nhead
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout

        self.embedding = nn.Embedding(self.ntoken, self.emsize)
        self.pos_encoder = PositionalEncoding(self.emsize, self.dropout)
        self.transformer = nn.Transformer(d_model=self.emsize, nhead=self.nhead, num_encoder_layers=self.nlayers, num_decoder_layers=self.nlayers)
        self.linear = nn.Linear(self.emsize, self.ntoken)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):    
        src = self.pos_encoder(self.embedding(src))
        tgt = self.pos_encoder(self.embedding(tgt))
        out = self.linear(self.transformer(src, tgt))
        out = F.log_softmax(out, dim=-1)
        return out