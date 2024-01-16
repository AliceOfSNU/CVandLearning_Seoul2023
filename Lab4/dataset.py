import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
import defines

import numpy as np 
import os
BASE_DIR = "../../data/{}"

class AudioDataset(Dataset):
    def __init__(self, split):
        # we will load all data at once.
        # memory issue?
        self.data = {
            "mfcc": [],
            "transcripts": []
        }
        self.PHONEMES = defines.PHONEMES
        self.SOS_TOKEN = 63
        self.EOS_TOKEN = 64
        base_dir = BASE_DIR.format(split)
        print("create dataset from data ", base_dir)
        
        # TODO load mfcc data from files
        mfcc_dir = os.path.join(base_dir, "mfcc")
        filecnt = 0
        for filename in os.listdir(mfcc_dir):
            full_filename = os.path.join(mfcc_dir, filename)
            with open(full_filename, "rb") as f:
                self.data["mfcc"].append(np.load(f))
            filecnt += 1
        
        self.length = filecnt
        print("\ttotal mfcc cnt: ", self.length)
        
        # TODO load transcript from files
        trans_dir = os.path.join(base_dir, "transcripts")
        filecnt = 0
        for filename in os.listdir(trans_dir):
            full_filename = os.path.join(trans_dir, filename)
            with open(full_filename, "rb") as f:
                line = np.load(f)
                # TODO convert line to int32.
                # maybe torch.Tensor conversion?
                self.data["transcripts"].append(
                    np.array([self.SOS_TOKEN] + [ord(tok) for tok in line[1:-1]] + [self.EOS_TOKEN])
                    )
            filecnt += 1
        
        self.length = filecnt
        print("\ttotal transcript cnt: ", self.length)
        
    def int_to_str(self, x):
        if x == 63: return '[SOS]'
        elif x == 64: return '[EOS]'
        else: return chr(x) #32, 65~
        
    def __getitem__(self, idx):
        return self.data["mfcc"][idx], self.data["transcripts"][idx]
    
    def __len__(self):
        return self.length
    
    def collate_fn(self, batch):
        '''
        TODO:
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish.
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features,
            and lengths of labels.
        '''
        # TODO: pad sequences and collect lengths
        mfccs, transcripts = [], []

        len_mfcc = []
        len_transcripts = []
        for mfcc, transcript in batch:
            mfccs.append(torch.as_tensor(mfcc))
            transcripts.append(torch.as_tensor(transcript))
            len_mfcc.append(len(mfcc))
            len_transcripts.append(len(transcript))
            
        padded_mfcc = pad_sequence(mfccs, batch_first=True)
        padded_transcripts = pad_sequence(transcripts, batch_first=True)
        
        # TODO: perform transforms
        
        return padded_mfcc, padded_transcripts, torch.tensor(len_mfcc), torch.tensor(len_transcripts)
    
#class AudioDatasetTest(Dataset):
    
# dataset for part 2 of hw4
# Memory Efficient in sense file loading happens in getitem to save RAM
class MEAudioDataset(Dataset):
    def __init__(self, split, transforms = None, cepstral=True):
        self.VOCAB      = defines.VOCAB
        self.cepstral   = cepstral

        print("create memory efficient dataset from data ", base_dir)
        
        # TODO load mfcc data from files
        if split == "train-clean-100" or split == "train-clean-360":
            base_dir = BASE_DIR.format(split)   
            mfcc_dir = os.path.join(base_dir, "mfcc")
            transcript_dir = os.path.join(base_dir, "transcripts")

            mfcc_files = os.listdir(mfcc_dir)
            transcript_files = os.listdir(transcript_dir)

        else:
            base_dir = BASE_DIR.format("train-clean-100")
            mfcc_dir = os.path.join(base_dir, "mfcc")
            transcript_dir = os.path.join(base_dir, "transcripts")

            mfcc_files = os.listdir(mfcc_dir)
            transcript_files = os.listdir(transcript_dir)

            base_dir = BASE_DIR.format("train-clean-360")
            mfcc_dir = os.path.join(base_dir, "mfcc")
            transcript_dir = os.path.join(base_dir, "transcripts")
            
            # add the list of mfcc and transcript paths from train-clean-360 to the list of paths  from train-clean-100
            mfcc_files.extend(os.listdir(mfcc_dir))
            transcript_files.extend(os.listdir(transcript_dir))

        assert len(mfcc_files) == len(transcript_files)
        self.length  = len(transcript_files)
        
        self.mfcc_files         = mfcc_files
        self.transcript_files   = transcript_files
        print("\ttotal mfcc cnt: ", len(self.mfcc_files))
        print("\ttotal transcript cnt: ", len(self.transcript_files))
        
    
    def __getitem__(self, idx):
        # read file
        return None
    
    def __len__(self):
        return self.length
    
    def collate_fn(self, batch):
        batch_x, batch_y, lengths_x, lengths_y = [], [], [], []

        for x, y in batch:
            # Add the mfcc, transcripts and their lengths to the lists created above
            # TODO
            pass

        # pack the mfccs and transcripts using the pad_sequence function from pytorch
        batch_x_pad =  None
        batch_y_pad =  None

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)