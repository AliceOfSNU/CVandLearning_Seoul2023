import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import random
import numpy as np
import defines
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

        #TODO: increase the number of channels
        self.embedding = nn.Conv1d(
            input_size,
            embedding_hidden_size,
            kernel_size=conv_kernel,
            stride=conv_stride,
            padding = conv_kernel//2
        )
        self.conv_stride = conv_stride
        self.conv_padding = conv_kernel//2
        self.conv_kernel = conv_kernel

        self.pBLSTMs = torch.nn.Sequential( 
            # TODO: Fill this up with pBLSTMs + locked dropouts
            pBLSTM(embedding_hidden_size, encoder_hidden_size),
            LockedDropout(dropout),
            # double because pblstm is bidirectional. 
            pBLSTM(2*encoder_hidden_size, encoder_hidden_size)
        )

    def forward(self, x, x_lens):
        #TODO: x
        x = x.transpose(-1, -2)
        x = self.embedding(x)
        x = x.transpose(-1, -2)
        
        #TODO: modify x_lens
        if self.conv_stride != 1:
            x_lens = 1 + (x_lens - self.conv_kernel + 2 * self.conv_padding)//self.conv_stride
            
        #TODO: pack
        x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        
        # TODO: pyramidal LSTM
        x = self.pBLSTMs(x)
        
        # TODO: unpack
        x, x_lens = pad_packed_sequence(x, batch_first=True)
        return x, x_lens
    
class MLPDecoder(torch.nn.Module):

    def __init__(self, embed_size, output_size= 41):
        super().__init__()

        self.fc1 = nn.Linear(embed_size, embed_size//2)
        self.bn = torch.nn.BatchNorm1d(embed_size//2)
        self.fc2 = nn.Linear(embed_size//2, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, encoder_out):
        #TODO call your MLP
        out = self.fc1(encoder_out)
        # apply batchnorm for each D-dimension.
        # D statistics are calculated over B*T.. 'time-averaging'+'batch-averaging'
        out = self.bn(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc2(F.relu(out))
        
        #TODO Think what should be the final output of the decoder for the classification
        # returns log-probability of 41 classes,
        # for each 'reduced-timestep'.
        out = self.softmax(out)
        return out
    
class ASRModel(torch.nn.Module):
    def __init__(self, input_size, embed_size= 192, output_size= len(defines.PHONEMES)):
        super().__init__()

        #self.augmentations  = torch.nn.Sequential(
        #    #TODO Add Time Masking/ Frequency Masking
        #    #Hint: See how to use PermuteBlock() function defined above
        #)
        self.encoder        = Encoder(input_size, embed_size, embed_size, conv_kernel=3, conv_stride=1, dropout=0.1)
        self.decoder        = MLPDecoder(2*embed_size, 41)

    def forward(self, x, lengths_x):

        #if self.training:
        #    x = self.augmentations(x)

        encoder_out, encoder_lens   = self.encoder(x, lengths_x)
        decoder_out                 = self.decoder(encoder_out)

        return decoder_out, encoder_lens