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

# run cofig
config = {
    "beam_width" : 3,
    "lr"         : 5e-4,
    "lr_schedule": "ReduceLROnPlateau with 3 patience",
    "epochs"     : 100,
    "batch_size": 96,
    "weight_decay" : 1e-3,
    "encoder_cnn_kernel": 3,
    "encoder_cnn_layers": 6,
    "encoder_cnn_channels": [64], #27 -> [/*here*/]
    "dropout": 0.1,
    "n_decodes": 10,
    "mlp_layers": 2,
    "tf_schedule": "Reduce by 0.1 every 15 epochs",
    "encoder_hidden_size": 64,
    "attn_hidden_size": 128,
    "decoder_hidden_size": 128,
    "data_features": 28,
    "init_method": "uniform -0.1~0.1",
    "run_id": "attn_64_residual_run1",
    "notes": "attention spellers with uniform init",
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
            shuffle=False,
            collate_fn=val_data.collate_fn
)
#test_loader = #TODO

model = ASRModel_Attention(
    input_size = 28, 
    encoder_hidden_size= 64,
    decoder_hidden_size= 128,
    intermediate_sizes = [64],
    output_size = len(defines.VOCAB)
)

optimizer =  torch.optim.AdamW(model.parameters(), config["lr"])
decoder = GreedyDecodeUtil()

# go
trainer = Trainer2(model, train_loader, val_loader,
                  decoder, config, verbose=True)
trainer.train()