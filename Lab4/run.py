from torch.utils.data import DataLoader
from dataset import AudioDataset
from model import ASRModel
import defines
train_data = AudioDataset('train-clean-100')
val_data = AudioDataset('dev-clean')
#test_data = AudioDatasetTest() #TODO

# run cofig
config = {
    'batch_size': 64,
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

model.forward(val_data[0])

