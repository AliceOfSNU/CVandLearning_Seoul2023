from torch.utils.data import DataLoader
from dataset import MEAudioDataset, verify_dataset
from model import ASRModel_Attention
import ctcdecode
import defines
import torch
import numpy as np
from trainer2 import Trainer2
import gc
from utils import GreedyDecodeUtil

torch.cuda.empty_cache()
gc.collect()

train_data = MEAudioDataset('train-clean-100',cepstral=True, base_dir="data/E2EASR_kaggle")
val_data = MEAudioDataset('dev-clean',cepstral=True, base_dir="data/E2EASR_kaggle")
#test_data = AudioDatasetTest() #TODO
verify_dataset(train_data, partition= 'train-clean-360')
verify_dataset(val_data, partition= 'dev-clean')

# run cofig
config = {
    "beam_width" : 3,
    "lr"         : 2e-4,
    "epochs"     : 100,
    "batch_size": 96,
    "weight_decay" : 5e-3,
    "encoder_cnn_kernel": 3,
    "encoder_cnn_layers": 6,
    "encoder_cnn_channels": [64], #27 -> [/*here*/]
    "dropout": 0.1,
    "mlp_layers": 2,
    "hidden_size": 64,
    "data_features": 28,
    "init_method": "default",
    "run_id": "attn_64_residual",
    "notes": "deeper conv1d embedding layers",
    "dataset": "E2EASR_kaggle/train-clean-100",
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

model = ASRModel_Attention(
    input_size = 28, 
    embed_size= 64,
    intermediate_sizes = [64],
    output_size = len(defines.VOCAB)
)
# Declare the decoder. Use the CTC Beam Decoder to decode phonemes
# CTC Beam Decoder Doc: https://github.com/parlance/ctcdecode
decoder = ctcdecode.CTCBeamDecoder(
    defines.LABELS,
    model_path=None,
    alpha=0,
    beta=0,
    cutoff_top_n=40,
    cutoff_prob=1.0,
    beam_width=3, #this should not exceed 50
    num_processes=4, #this should be ~#of cpus
    blank_id=0,
    log_probs_input=True
)


optimizer =  torch.optim.AdamW(model.parameters(), config["lr"])
decoder = GreedyDecodeUtil()
tf_schedule = 1.0 # modify this value -> do not go below 0.5


# go
trainer = Trainer2(model, train_loader, val_loader,
                  decoder, config, verbose=True)
trainer.train()