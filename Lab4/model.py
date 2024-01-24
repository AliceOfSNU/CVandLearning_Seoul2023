import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.distributions.categorical import Categorical
from torchaudio.transforms import FrequencyMasking, TimeMasking
import random
import numpy as np
import defines
import math
import gc

# third party
try:
    import zipfile
    import pandas as pd
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score
    import ctcdecode
    from ctcdecode import CTCBeamDecoder
    import Levenshtein
except:
    print("There is one or more dependency not met.")
    
class pBLSTM(torch.nn.Module):

    '''
    Pyramidal BiLSTM
    from Listen, Attend, and Spell paper
    https://arxiv.org/pdf/1508.01211.pdf
    
    The 'Listen' part of LAS. downsamples in time axis
    to cope with long inputs and extract important features from them.
    '''

    def __init__(self, input_size, hidden_size):
        super(pBLSTM, self).__init__()
        # TODO: single layer bidirectional LSTM
        self.blstm = nn.LSTM(
            # double input_size because trunc_reshape doubles feature dim
            2*input_size, hidden_size, num_layers = 1, bidirectional=True,
        )

    def forward(self, x_packed): # x_packed is a PackedSequence

        # TODO: Pad Packed Sequence
        x_pad, x_lens = pad_packed_sequence(x_packed, batch_first=True)
        # Call self.trunc_reshape() which downsamples the time steps of x and increases the feature dimensions as mentioned above
        # self.trunc_reshape will return 2 outputs. What are they? Think about what quantites are changing.
        x_pad, x_lens = self.trunc_reshape(x_pad, x_lens)
        
        # TODO: Pack Padded Sequence. What output(s) would you get?
        x_packed = pack_padded_sequence(x_pad, x_lens, batch_first=True, enforce_sorted=False)
        
        # TODO: Pass the sequence through bLSTM
        output_packed, (hn, cn) = self.blstm(x_packed)
        # output_packed has .data, .batch_size, .sorted_indicies..
        # output_packed.data.shape = (B * T, 2 * Hidden_dim), 2* because two directions are concat in last dim
        # hn and cn contains only the last hidden/cell states.
        # we need output for each timestep, packed.
        
        return output_packed

    def trunc_reshape(self, x, x_lens):        
        # TODO: Reshape x. 
        B, T, D = x.shape
        if T % 2 == 1:
            x = x[:, :-1, :] # discard last
        x = x.reshape(B, T//2, 2 * D) #operates on padded tensor, with all the zeros.
        
        # instead, modify lens to len//2 
        # for odd T like T=5 -> new T=2. which means (0, 1), (2, 3) and 4 is discarded
        # the modified lengths is to be provided to pack_padded_tensor of next stage
        return x, x_lens//2
    

class LockedDropout(nn.Module):
    def __init__(self, dropout=0.2):
        self.dropout = dropout
        super().__init__()

    def forward(self, x):
        if not self.training:
            return x
        
        # TODO: unpack
        x, lens = pad_packed_sequence(x, batch_first=False)
        
        # TODO: apply same dropout over all timesteps.
        # choose random dropout mask for each batch.
        m = x.data.new(1, *x.shape[1:]).bernoulli_(1 - self.dropout)
        m.requires_grad = False
        mask = m / (1 - self.dropout)
        mask = mask.expand_as(x)
        
        # TODO: pack again and return
        return pack_padded_sequence(mask * x, lens, batch_first=False, enforce_sorted=False)

class ResidualBlock(nn.Module):
    def __init__ (self, num_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, num_channels, kernel_size, padding = kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.activ1 = nn.ReLU()
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size, padding = kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_channels)
        self.activ2 = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activ1(out) #bn is done for each channel
        out = self.conv2(out)
        out = self.bn2(out)
        return self.activ2(out + x)
    
class Encoder(torch.nn.Module):
    '''
    The Encoder takes utterances as inputs and returns latent feature representations
    '''
    def __init__(self, input_size, encoder_hidden_size, embedding_hidden_size,
        conv_kernel = 3, #3 or 5.
        conv_stride = 2, #reduces length in time dim.
        dropout = 0.1,
                 ):
        super(Encoder, self).__init__()

        #increase the number of channels
        #self.embedding = nn.Conv1d(
        #    input_size,
        #    embedding_hidden_size,
        #    kernel_size=conv_kernel,
        #    stride=conv_stride,
        #    padding = conv_kernel//2
        #)
        
        self.embedding = nn.Sequential(
            nn.Conv1d(input_size, embedding_hidden_size, kernel_size=conv_kernel, stride=conv_stride, padding = conv_kernel//2),
            nn.BatchNorm1d(embedding_hidden_size),
            nn.ReLU(),
            ResidualBlock(embedding_hidden_size, kernel_size = conv_kernel),
        )
        self.conv_stride = conv_stride
        self.conv_padding = conv_kernel//2
        self.conv_kernel = conv_kernel

        self.lstm = nn.LSTM(
            embedding_hidden_size, encoder_hidden_size, 
            num_layers = 1, bidirectional=True
        )
        self.pBLSTMs = nn.Sequential( 
            # TODO: Fill this up with pBLSTMs + locked dropouts
            pBLSTM(2*encoder_hidden_size, encoder_hidden_size),
            LockedDropout(dropout),
            # double because pblstm is bidirectional. 
            pBLSTM(2*encoder_hidden_size, encoder_hidden_size)
        )

    def forward(self, x, x_lens):
        #channel first layout. time dim goes last
        x = x.transpose(-1, -2)
        x = self.embedding(x)
        x = x.transpose(-1, -2)
        
        #modify x_lens
        if self.conv_stride != 1:
            x_lens = 1 + (x_lens - self.conv_kernel + 2 * self.conv_padding)//self.conv_stride
            
        #pack
        x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        
        #pyramidal LSTM
        x, (hn, cn) = self.lstm(x)
        x = self.pBLSTMs(x)
        
        #unpack
        x, x_lens = pad_packed_sequence(x, batch_first=True)
        return x, x_lens
    
class DotProductAttention(torch.nn.Module):
    def __init__(self, dim_q, dim_kv, hidden_dim, dropout=0.1):
        super(DotProductAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(dim_q, hidden_dim)
        self.key_proj = nn.Linear(dim_kv, hidden_dim)
        self.value_proj = nn.Linear(dim_kv, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.cache = {}
        
    # precompute k and v, to reuse them for multiple queries.
    def compute_kv(self, x):
        self.cache["k"] = self.key_proj(x)
        self.cache["v"] = self.value_proj(x)
    
    def compute_context(self, q, need_weights = False):
        q = self.query_proj(q)
        
        if "k" not in self.cache.keys() or "v" not in self.cache.keys():
            print("[err] compute_kv must be called before calling compute_context")
        
        dot_product = torch.matmul(q, self.cache["k"].transpose(-2, -1))
        attn = dot_product / math.sqrt(self.hidden_dim)
        attn = self.dropout(attn)
        context_vec = torch.matmul(attn, self.cache["v"])
        
        if need_weights:
            return context_vec, attn
        else:
            return context_vec
        
class MLPDecoder(torch.nn.Module):

    def __init__(self, embed_size, output_size= 41):
        super(MLPDecoder, self).__init__()

        self.fc1 = nn.Linear(embed_size, embed_size//2)
        self.bn = nn.BatchNorm1d(embed_size//2)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(embed_size//2, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, encoder_out):
        # call your MLP
        out = self.fc1(encoder_out)
        # apply batchnorm for each D-dimension.
        # D statistics are calculated over B*T.. 'time-averaging'+'batch-averaging'
        out = self.bn(out.transpose(1, 2)).transpose(1, 2)
        out = self.activation(out)
        out = self.fc2(out)
        
        # Think what should be the final output of the decoder for the classification
        # returns log-probability of 41 classes,
        # for each 'reduced-timestep'.
        out = self.softmax(out)
        return out
    
class AttentionDecoder(torch.nn.Module):
    def __init__(self, attender:DotProductAttention, hidden_dim, output_dim, dropout = 0.1):
        super(AttentionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.attender = attender # Attention object in speller
        #self.max_timesteps = # Max timesteps

        self.embedding =  nn.Embedding(128, hidden_dim)
        self.lstm_cell1 = nn.LSTMCell(hidden_dim, hidden_dim)
        # you may want to insert a dropout here!
        self.lstm_cell2 = nn.LSTMCell(hidden_dim, hidden_dim)

        # For CDN (Feel free to change)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.CDN = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.output_dim = output_dim
        #self.char_prob.weight = # Weight tying (From embedding layer)

    def forward(self, y, teacher_forcing=True):
        B, T = y.shape

        output_idxs = torch.empty((B, 0,)) # cumulate choices over timesteps
        output_logits = torch.empty((B, 0, self.output_dim))
        attentions = [] #cumulate attentions for later debugging
        
        yD = self.embedding(y) #B, T, D
        
        hc1 = (torch.zeros((B, self.hidden_dim)), torch.zeros((B, self.hidden_dim)))
        hc2 = (torch.zeros((B, self.hidden_dim)), torch.zeros((B, self.hidden_dim)))
        for t in range(T):
            if teacher_forcing:
                x = yD[:, t, :] #B, D
            else:
                x = self.embedding(out)
            hc1 = self.lstm_cell1(x, hc1)
            hc2 = self.lstm_cell2(hc1[0], hc2)
            ctx, attn = self.attender.compute_context(hc2[0].unsqueeze(1), need_weights=True)
            attentions.append(attn)
            out = self.softmax(self.CDN(torch.cat([hc2[0], ctx.squeeze(1)], dim = -1)))
            output_logits = torch.cat([output_logits, out.unsqueeze(1)], dim=1)
            # decodes a sample.. (using random strategy)
            out = Categorical(logits=out).sample().unsqueeze(-1)
            output_idxs = torch.cat([output_idxs, out], dim=1)
            
        attentions = torch.cat(attentions, dim = 1) # B * T(=decode timesteps) * S(=kv length from encoder)
        return output_logits, attentions
        
class ASRModel(torch.nn.Module):
    def __init__(self, input_size, embed_size= 192, 
                 conv_kernel = 3, 
                 output_size= len(defines.PHONEMES),
                 dropout = 0.1):
        super().__init__()

        self.augmentations  = torch.nn.Sequential(
            #Add Time Masking/ Frequency Masking
            FrequencyMasking(freq_mask_param=6),
            TimeMasking(time_mask_param=80),
        )
        self.encoder        = Encoder(input_size, embed_size, embed_size, conv_kernel=conv_kernel, conv_stride=1, dropout=dropout)
        self.decoder        = MLPDecoder(2*embed_size, 41)

    def forward(self, x, lengths_x):

        if self.training:
            x = self.augmentations(x.transpose(-1, -2)).transpose(-1, -2)

        encoder_out, encoder_lens   = self.encoder(x, lengths_x)
        decoder_out                 = self.decoder(encoder_out)

        return decoder_out, encoder_lens
    
class ASRModel_Attention(torch.nn.Module):
    def __init__(self, input_size, embed_size, output_size= len(defines.PHONEMES)): # add parameters
        super().__init__()

        # Pass the right parameters here
        self.listener = Encoder(input_size, embed_size, embed_size, conv_kernel=3, conv_stride=1, dropout=0.1)
        self.attend = DotProductAttention(
            dim_q=embed_size, dim_kv=2*embed_size, hidden_dim=embed_size, dropout=0.1
        )
        self.speller = AttentionDecoder(self.attend, embed_size, output_size)
        self.output_size = output_size    
    
    def compute_loss(self, dist, y):
        # dist(B * T * Vocab_size): logits
        # y(B * T)int64 indexes
        y_onehot = F.one_hot(y, self.output_size)
        return  -(dist* y_onehot).sum()
        
    def forward(self, x,lx,y=None, teacher_forcing_ratio=1):
        # Encode speech features
        encoder_outputs, encoder_lens = self.listener(x,lx)

        # We want to compute keys and values ahead of the decoding step, as they are constant for all timesteps
        # Set keys and values using the encoder outputs
        self.attend.compute_kv(encoder_outputs)

        # Decode text with the speller using context from the attention
        outputs, attentions = self.speller(y, teacher_forcing=True)
        return outputs, attentions