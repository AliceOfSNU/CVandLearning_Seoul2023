from model import ASRModel_Attention
from dataset import MEAudioDatasetTest, MEAudioDataset
import torch
from defines import VOCAB
from utils import indices_to_chars, load_model
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Levenshtein
import seaborn as sns

test_data = MEAudioDatasetTest(False,  base_dir="data/E2EASR_kaggle")
val_data = MEAudioDataset('dev-clean',cepstral=True, base_dir="data/E2EASR_kaggle")

np.random.seed(12)
torch.manual_seed(12)

test_loader     = torch.utils.data.DataLoader(
    dataset     = test_data,
    batch_size  = 96, #test set has large mfcc lengths. reduce this?
    shuffle     = False,
    pin_memory  = True,
    collate_fn  = test_data.collate_fn
)

val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=96,
            drop_last=False,
            shuffle=False,
            collate_fn=val_data.collate_fn
)
device = 'cuda' if torch.cuda.is_available() else "cpu"

model = ASRModel_Attention(
    input_size = 28, 
    encoder_hidden_size= 64,
    decoder_hidden_size= 128,
    intermediate_sizes = [64],
    output_size = len(VOCAB)
)

base_dir = "CVandLearning_Seoul2023/Lab4"
model_dir = os.path.join(base_dir, 'model/attn_64_tf_from_0.9_linear')

model, _, epoch, val_loss = load_model(os.path.join(model_dir, "epoch95.pt"), model)
model.to(device)

def test():
    for i, data in enumerate(test_loader):
        x, lx = data
        x = x.to(device) 
        lx= lx.to(device)

        with torch.inference_mode():
            decodes, log_probs = model.generate_decodes(10, x, lx, y=None, teacher_forcing_ratio=0.0)
            pred_strings = [indices_to_chars(line, VOCAB) for line in decodes.cpu().numpy()]
            
def test_with_valid_set(write_csv=False):
    out_dir = os.path.join(model_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)
    df_batch = []
    for i, data in enumerate(val_loader):
        x, y, lx, _ = data
        x = x.to(device)
        with torch.inference_mode():
            # change the number of decoder runs!
            decodes, log_probs = model.generate_decodes(100, x, lx, y=None, teacher_forcing_ratio=0.0)
            pred_strings = [indices_to_chars(line, VOCAB) for line in decodes.cpu().numpy()]
            label_strings = [indices_to_chars(line, VOCAB) for line in y.cpu().numpy()]
            lv_dists = [Levenshtein.distance(pred, gt) for pred, gt in zip(pred_strings, label_strings)]
                
        df_batch.append(pd.DataFrame({'pred': pred_strings, 'label':label_strings,
                        'log_prob': log_probs.cpu().numpy(), 'lev_dist': lv_dists}))
        
        if i == 2: break # not doing this forever..

    # combine results and write to csv
    df = pd.concat(df_batch, ignore_index=True)
    # print some stats
    print("mean {:.04f}\tstd {:.04f}\tmin {:.04f}\tmax {:.04f}"
            .format(df['lev_dist'].mean(), df["lev_dist"].std(),
                    df['lev_dist'].min(), df['lev_dist'].max()))
    df.to_csv(os.path.join(out_dir, 'valid.csv'), index=False)
    sns.scatterplot(data=df, x="lev_dist", y="log_prob").set(title='lev_dist(x) vs log_prob(y)')
    plt.savefig(os.path.join(out_dir, "seaborn_plot.png"))
    
test_with_valid_set(write_csv=True)