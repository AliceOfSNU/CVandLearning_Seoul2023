from torch.utils.data import DataLoader
from dataset import AudioDataset
from model import ASRModel
import ctcdecode
import defines
from trainer import Trainer

train_data = AudioDataset('train-clean-100', data_dir="data/ARPAbet_kaggle")
val_data = AudioDataset('dev-clean', data_dir="data/ARPAbet_kaggle")
#test_data = AudioDatasetTest() #TODO

# run cofig
config = {
    "beam_width" : 2,
    "lr"         : 2e-3,
    "epochs"     : 50,
    "batch_size": 64,
    "run_id": 1
}

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
    input_size = 28, 
    embed_size= 64,
    output_size = len(defines.PHONEMES)
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

# go
trainer = Trainer(model, train_loader, val_loader,
                  decoder, config, verbose=True)
trainer.train()