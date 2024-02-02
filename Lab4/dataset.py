import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import defines

import numpy as np 
import os
BASE_DIR = "../../data"

class AudioDataset(Dataset):
    def __init__(self, split, data_dir = BASE_DIR):
        # we will load all data at once.
        # memory issue?
        self.data = {
            "mfcc": [],
            "transcripts": []
        }
        self.PHON2ALPHA = defines.CMUdict_ARPAbet
        self.PHONEMES = defines.PHONEMES
        self.LABELS = defines.LABELS
        self.LABEL2IDX = { v: k for k, v in enumerate(self.LABELS)}
        
        base_dir = os.path.join(data_dir, split)
        print("create dataset from data ", base_dir)
        
        #load mfcc data from files
        mfcc_dir = os.path.join(base_dir, "mfcc")
        filecnt = 0
        for filename in os.listdir(mfcc_dir):
            full_filename = os.path.join(mfcc_dir, filename)
            with open(full_filename, "rb") as f:
                data = np.load(f)
                # normalize data here
                data = data[:, :27]
                data = data - data.mean(axis = 0, keepdims=True) #normalize along time axis
                #
                self.data["mfcc"].append(data)
            filecnt += 1
        
        self.length = filecnt
        print("\ttotal mfcc cnt: ", self.length)
        
        # load transcript from files
        trans_dir = os.path.join(base_dir, "transcript")
        filecnt = 0
        for filename in os.listdir(trans_dir):
            full_filename = os.path.join(trans_dir, filename)
            with open(full_filename, "rb") as f:
                line = np.load(f)
                # convert line to intTensor, 
                # first map to char's by CMUdict_ARPAbet and convert to int
                # SOS and EOS are dropped!
                self.data["transcripts"].append(
                    np.array([self.LABEL2IDX[self.PHON2ALPHA[tok]] for tok in line[1:-1]])
                    )
            filecnt += 1
        
        assert filecnt == self.length, "number of audio files and transcipt files should match"
        
        self.length = filecnt
        print("\ttotal transcript cnt: ", self.length)
        
    def idx_to_str(self, x):
        # input a idx and returns the corresponding phoneme
        #   x Integer: index
        return self.PHONEMES[x] #32, 65~
        
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
            transcripts.append(torch.as_tensor(transcript, dtype=torch.long))
            len_mfcc.append(len(mfcc))
            len_transcripts.append(len(transcript))
            
        padded_mfcc = pad_sequence(mfccs, batch_first=True)
        padded_transcripts = pad_sequence(transcripts, batch_first=True)
        
        # TODO: perform transforms
        
        return padded_mfcc, padded_transcripts, torch.tensor(len_mfcc, dtype=torch.long), torch.tensor(len_transcripts, dtype=torch.long)
    
#class AudioDatasetTest(Dataset):
    
# dataset for part 2 of hw4
# Memory Efficient in sense file loading happens in getitem to save RAM
class MEAudioDataset(Dataset):
    def __init__(self, split, transforms = None, cepstral=True, base_dir = BASE_DIR):
        self.VOCAB      = defines.VOCAB
        self.VOCAB_MAP = defines.VOCAB_MAP
        self.cepstral   = cepstral

        print("create memory efficient dataset from data ", os.path.join(base_dir, split))
        
        # load mfcc data from files
        if split == "train-clean-100" or split == "train-clean-360" or split == "dev-clean":
            mfcc_dir = os.path.join(base_dir, split, "mfcc")
            transcript_dir = os.path.join(base_dir, split, "transcripts")

            mfcc_files = [os.path.join(mfcc_dir, fname) for fname in os.listdir(mfcc_dir)]
            transcript_files = [os.path.join(transcript_dir, fname)for fname in os.listdir(transcript_dir)]

        else: #both 100 + 360 == 460 dataset
            mfcc_dir = os.path.join(base_dir, "train-clean-100", "mfcc")
            transcript_dir = os.path.join(base_dir, "train-clean-100", "transcripts")

            mfcc_files = [os.path.join(mfcc_dir, fname) for fname in os.listdir(mfcc_dir)]
            transcript_files = [os.path.join(transcript_dir, fname) for fname in os.listdir(transcript_dir)]

            mfcc_dir = os.path.join(base_dir, "train-clean-360", "mfcc")
            transcript_dir = os.path.join(base_dir, "train-clean-360", "transcripts")
            
            # add the list of mfcc and transcript paths from train-clean-360 to the list of paths  from train-clean-100
            files = os.listdir(mfcc_dir)
            files = os.listdir(transcript_dir)
            mfcc_files_360 = [os.path.join(mfcc_dir, fname) for fname in files[:]]
            transcript_files_360 = [os.path.join(transcript_dir, fname) for fname in files[:]]
            for idx in range(len(mfcc_files_360)):
                with open(self.mfcc_files360[idx], "rb") as f:
                    mfcc = np.load(f)
                    if mfcc.shape[0] > 2448: continue
                    mfcc_files.append(mfcc_files_360[idx])
                    transcript_files.append(transcript_files_360[idx])
                    
        assert len(mfcc_files) == len(transcript_files)
        self.length  = len(transcript_files)
        
        self.mfcc_files         = mfcc_files
        self.transcript_files   = transcript_files
        print("\ttotal mfcc cnt: ", len(self.mfcc_files))
        print("\ttotal transcript cnt: ", len(self.transcript_files))
        
    
    def __getitem__(self, idx):
        # read file
        with open(self.mfcc_files[idx], "rb") as f:
            mfcc = np.load(f)
        
        with open(self.transcript_files[idx], "rb") as f:
            transcript = np.load(f)
            # remove SOS so we can compare directly to outputs
            transcript = np.array([self.VOCAB_MAP[tok] for tok in transcript[1:]])
            
        return mfcc, transcript
    
    def __len__(self):
        return self.length
    
    def collate_fn(self, batch):
        mfccs, transcripts, mfcc_lens, transcript_lens = [], [], [], []

        for mfcc, transcript in batch:
            mfccs.append(torch.as_tensor(mfcc))
            transcripts.append(torch.as_tensor(transcript, dtype=torch.long))
            mfcc_lens.append(len(mfcc))
            transcript_lens.append(len(transcript))
            
        padded_mfcc = pad_sequence(mfccs, batch_first=True)
        padded_transcripts = pad_sequence(transcripts, batch_first=True)

        return padded_mfcc, padded_transcripts, torch.tensor(mfcc_lens, dtype=torch.long), torch.tensor(transcript_lens, dtype=torch.long)
    
    
# dataset checker
def verify_dataset(dataset, partition= 'train-clean-100'):
    print("\nPartition loaded     : ", partition)
    if partition != 'test-clean':
        print("Max mfcc length          : ", np.max([data[0].shape[0] for data in dataset]))
        print("Avg mfcc length          : ", np.mean([data[0].shape[0] for data in dataset]))
        print("Max transcript length    : ", np.max([data[1].shape[0] for data in dataset]))
        print("Max transcript length    : ", np.mean([data[1].shape[0] for data in dataset]))
    else:
        print("Max mfcc length          : ", np.max([data.shape[0] for data in dataset]))
        print("Avg mfcc length          : ", np.mean([data.shape[0] for data in dataset]))


