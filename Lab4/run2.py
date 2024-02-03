from torch.utils.data import DataLoader
from dataset import MEAudioDataset, MEAudioDatasetTest, verify_dataset
from model import ASRModel_Attention
import ctcdecode
import defines
import torch
import numpy as np
from trainer2 import Trainer2
import gc
import os
from utils import GreedyDecodeUtil
import json

torch.cuda.empty_cache()
gc.collect()

BASE_DIR= "CVandLearning_Seoul2023/Lab4"
run_id = "attn_128"
MODEL_DIR = os.path.join(BASE_DIR, "model", run_id)
DATA_DIR = "data/E2EASR_kaggle"
with open(os.path.join(BASE_DIR, "config.json"), "r") as f:
    config = json.load(f)
    config["run_id"] = run_id

train_data = MEAudioDataset('train-clean-100',cepstral=True, base_dir=DATA_DIR)
val_data = MEAudioDataset('dev-clean',cepstral=True, base_dir=DATA_DIR)
#test_data = MEAudioDatasetTest('test-clean',cepstral=True, base_dir=DATA_DIR)

np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

# Do NOT forget to pass in the collate function as parameter while creating the dataloader
train_loader = DataLoader(
            train_data,
            batch_size=config["batch_size"],
            drop_last=True,
            shuffle=True,
            collate_fn=train_data.collate_fn
)
val_loader = DataLoader(
            val_data,
            batch_size=config["batch_size"],
            drop_last=False,
            shuffle=False,
            collate_fn=val_data.collate_fn
)
#test_loader = DataLoader(
#            test_data,
#            batch_size=config["batch_size"],
#            drop_last=False,
#            shuffle=False,
#            collate_fn=test_data.collate_fn
#)

model = ASRModel_Attention(
    input_size = config["data_features"], 
    encoder_hidden_size= config["encoder_hidden_size"],
    decoder_hidden_size= config["decoder_hidden_size"],
    intermediate_sizes = config["encoder_cnn_channels"],
    output_size = len(defines.VOCAB),
    dropout = config["dropout"]
)

optimizer =  torch.optim.AdamW(model.parameters(), config["lr"])
decoder = GreedyDecodeUtil()

# go
trainer = Trainer2(model, train_loader, val_loader,
                  decoder, config, verbose=True)
trainer.train()