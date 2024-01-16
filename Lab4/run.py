from torch.utils.data import DataLoader
from dataset import AudioDataset
from model import ASRModel
import ctcdecode
import defines
import torch

train_data = AudioDataset('train-clean-100')
val_data = AudioDataset('dev-clean')
#test_data = AudioDatasetTest() #TODO

# run cofig
config = {
    "beam_width" : 2,
    "lr"         : 2e-3,
    "epochs"     : 50,
    "batch_size": 64,
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
    input_size = 27, 
    embed_size= 64,
    output_size = len(defines.PHONEMES)
)

criterion = torch.nn.CTCLoss()
optimizer =  torch.optim.AdamW(model.parameters(), config["lr"]) # What goes in here?

# Declare the decoder. Use the CTC Beam Decoder to decode phonemes
# CTC Beam Decoder Doc: https://github.com/parlance/ctcdecode
decoder = ctcdecode.CTCBeamDecoder(
    labels,
    model_path=None,
    alpha=0,
    beta=0,
    cutoff_top_n=40,
    cutoff_prob=1.0,
    beam_width=32, #this should not exceed 50
    num_processes=4, #this should be ~#of cpus
    blank_id=0,
    log_probs_input=True
)

# test impl
model.eval()
device = 'cuda' if torch.cuda.is_available() else "cpu"

for i, data in enumerate(val_loader, 0):
    x, y, lx, ly = data
    x, y = x.to(device), y.to(device)
    h, lh = model(x, lx)
    h = torch.permute(h, (1, 0, 2)) #B, T, D
    
    optimizer.zero_grad()
    #note h is in time-first layout
    #h = (T, B, D) -> log_probs, y = (B, S) -> indexes
    #lh = (B), ly = (B)
    loss = criterion(h, y, lh, ly)
    print(loss)
    loss.backward()
    optimizer.step()
    
    #print(calculate_levenshtein(h, y, lx, ly, decoder, LABELS))
    break