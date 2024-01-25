from torch.utils.data import DataLoader
from dataset import AudioDataset
from model import ASRModel
import defines
from utils import CTCDecodeUtil, GreedyDecodeUtil
from trainer import Trainer
import numpy as np
import torch

train_data = AudioDataset('train-clean-100', data_dir="data/ARPAbet_kaggle")
val_data = AudioDataset('dev-clean', data_dir="data/ARPAbet_kaggle")
#test_data = AudioDatasetTest() #TODO
# run cofig
config = {
    "beam_width" : 3,
    "lr"         : 2e-3,
    "epochs"     : 50,
    "batch_size": 64,
    "encoder_cnn_kernel": 3,
    "encoder_cnn_layers": 6,
    "encoder_cnn_channels": [64], #27 -> [/*here*/]
    "dropout": 0.1,
    "mlp_layers": 2,
    "hidden_size": 64,
    "data_features": 27,
    "init_method": "xavier_uniform",
    "run_id": "base_64_residual_27data_w_augs_xavier",
    "notes": "deeper conv1d embedding layers",
    "dataset": "ARPAbet_kaggle/train-clean-100",
    "seed":12
}

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
            shuffle=True,
            collate_fn=val_data.collate_fn
)
#test_loader = #TODO

model = ASRModel(
    input_size = 27, 
    embed_size= config["hidden_size"],
    intermediate_sizes = config["encoder_cnn_channels"],
    conv_kernel = config["encoder_cnn_kernel"],
    dropout= config["dropout"],
    output_size = len(defines.PHONEMES)
)
# Declare the decoder. Use the CTC Beam Decoder to decode phonemes
# CTC Beam Decoder Doc: https://github.com/parlance/ctcdecode
decoder = CTCDecodeUtil(beam_width=3)

# go
trainer = Trainer(model, train_loader, val_loader,
                  decoder, config, verbose=True)
trainer.train()