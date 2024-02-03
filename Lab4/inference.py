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
import json
test_data = MEAudioDatasetTest(False,  base_dir="data/E2EASR_kaggle")
val_data = MEAudioDataset('dev-clean',cepstral=True, base_dir="data/E2EASR_kaggle")

BASE_DIR= "CVandLearning_Seoul2023/Lab4"
run_id = "attn_64_loop_context"
MODEL_DIR = os.path.join(BASE_DIR, "model", run_id)
DATA_DIR = "data/E2EASR_kaggle"
with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
    config = json.load(f)
    config["run_id"] = run_id
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

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
    input_size = config["data_features"], 
    encoder_hidden_size= config["encoder_hidden_size"],
    decoder_hidden_size= config["decoder_hidden_size"],
    intermediate_sizes = config["encoder_cnn_channels"],
    output_size = len(VOCAB),
    dropout = config["dropout"]
)
model, _, epoch, val_loss = load_model(os.path.join(MODEL_DIR, "epoch95.pt"), model)
model.to(device)
model.eval()

def test():
    for i, data in enumerate(test_loader):
        x, lx = data
        x = x.to(device) 
        lx= lx.to(device)

        with torch.inference_mode():
            decodes, log_probs = model.generate_decodes(10, x, lx, y=None, teacher_forcing_ratio=0.0)
            pred_strings = [indices_to_chars(line, VOCAB) for line in decodes.cpu().numpy()]
            
def test_with_valid_set(mode = "greedy"):
    out_dir = os.path.join(MODEL_DIR, 'output')
    os.makedirs(out_dir, exist_ok=True)
    df_batch = []
    for i, data in enumerate(val_loader):
        x, y, lx, _ = data
        x = x.to(device)
        with torch.inference_mode():
            # change the number of decoder runs!
            if mode == "greedy":
                decodes, attns = model.generate_greedy(x, lx)
            elif mode == "random":
                decodes, log_probs = model.generate_decodes(100, x, lx, y=None, teacher_forcing_ratio=0.0)
                
            pred_strings = [indices_to_chars(line, VOCAB) for line in decodes.cpu().numpy()]
            label_strings = [indices_to_chars(line, VOCAB) for line in y.cpu().numpy()]
            lv_dists = [Levenshtein.distance(pred, gt) for pred, gt in zip(pred_strings, label_strings)]
            dic  = {'pred': pred_strings, 'label':label_strings, 'lev_dist': lv_dists}
            
            if mode == "random":
                dic['log_prob']: log_probs.cpu().numpy()
        df_batch.append(pd.DataFrame(dic))
        
        if i == 2: break # not doing this forever..

    # combine results and write to csv
    df = pd.concat(df_batch, ignore_index=True)
    # print some stats
    print("mean {:.04f}\tstd {:.04f}\tmin {:.04f}\tmax {:.04f}"
            .format(df['lev_dist'].mean(), df["lev_dist"].std(),
                    df['lev_dist'].min(), df['lev_dist'].max()))
    df.to_csv(os.path.join(out_dir, 'valid.csv'), index=False)
    
    # plot needed stuff
    #sns.scatterplot(data=df, x="lev_dist", y="log_prob").set(title='lev_dist(x) vs log_prob(y)')
    sns.histplot(data=df, x="lev_dist")
    plt.savefig(os.path.join(out_dir, "seaborn_plot.png"))
    
test_with_valid_set("greedy")