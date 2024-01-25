import torch
from tqdm import tqdm
from utils import calculate_levenshtein, indices_to_chars
from defines import VOCAB, PAD_TOKEN
import os
import wandb
import matplotlib.pyplot as plt
USE_WANDB = True
BASE_DIR= "CVandLearning_Seoul2023/Lab4"

class Trainer2():
    # bsaic trainer class.
    def __init__(self, model, train_loader, val_loader, 
                 decoder, # either CTCDecodeUtil, GreedyDecodeUtil,
                 config, verbose=True):
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.verbose = verbose
        if verbose:
            print("using device.. ", self.device)
            print(self.model.named_modules)
        self.model.to(self.device)
        self.decoder = decoder
        
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.run_id = config["run_id"]
        config.pop("run_id")
        os.makedirs(os.path.join(BASE_DIR, f"model/{self.run_id}"), exist_ok=True)
        if USE_WANDB:
            self.run = wandb.init(
                name = self.run_id, 
                reinit = True, 
                # run_id = ### Insert specific run id here if you want to resume a previous run
                # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
                project = "LAS_asr", 
                config=config
            )
            
        self.lr = config["lr"]
        self.nepochs = config["epochs"]
        self.criterion = torch.nn.CrossEntropyLoss(
                ignore_index=PAD_TOKEN, #ignore paddings
                reduction='mean',
            )
        self.optimizer =  torch.optim.AdamW(model.parameters(), config["lr"]) # What goes in here?
        self.lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', threshold=0.001, factor = 0.2, patience = 2)
        
    def train(self):
        best_loss = 10e6
        for epoch in range(self.nepochs):
            self.model.train()
            print(f"Epoch {epoch}/{self.nepochs}")
            batch_bar = tqdm(total=len(self.train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
            epoch_loss = 0
            for i, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                x, y, lx, ly = data
                x, y = x.to(self.device), y.to(self.device)

                h, attn = self.model(x, lx, y)
                loss = self.criterion(h.transpose(-1, -2), y)

                epoch_loss += loss.item()
                
                ## add batch logs here
                batch_bar.set_postfix(
                    loss="{:.04f}".format(epoch_loss / (i + 1)),
                    lr="{:.06f}".format(float(self.optimizer.param_groups[0]['lr'])))

                batch_bar.update() # Update tqdm bar
                ## ^ add batch log here ^
                
                # step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # cleanup
                del x, y, lx, ly, h, loss
                torch.cuda.empty_cache()
                
                ### TESTING ONLY.
                #if i == 2: break

            batch_bar.close() 
            train_loss = epoch_loss / len(self.train_loader)
            print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, float(self.optimizer.param_groups[0]['lr'])))
            
            # validate
            valid_loss, valid_dist = self.validate(epoch)
            self.lr_schedule.step(valid_dist) #step lr decay
            
            ## checkpointing
            if valid_loss < best_loss:
                if self.verbose: print("best loss!")
                best_loss = valid_loss
                self.save_model(epoch, valid_loss, os.path.join(BASE_DIR, f"model/{self.run_id}/best.pt"))
            if epoch % 5 == 0:
                self.save_model(epoch, valid_loss, os.path.join(BASE_DIR, f"model/{self.run_id}/epoch{epoch}.pt"))
            ## checkpointing ends
            
            ## add epoch logs here
            print("\tVal Dist {:.04f}%\t Val Loss {:.04f}".format(valid_dist, valid_loss))
            if USE_WANDB:
                wandb.log({
                    'train_loss': train_loss,
                    'valid_dist': valid_dist,
                    'valid_loss': valid_loss,
                    'lr'        : float(self.optimizer.param_groups[0]['lr'])
                })
            ## ^ add epoch logs here ^
        if USE_WANDB:
            self.run.finish() 
        return 

    def validate(self, epoch, phoneme_map=VOCAB):
        self.model.eval()
        batch_bar = tqdm(total=len(self.val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

        total_loss = 0
        vdist = 0

        for i, data in enumerate(self.val_loader):
            x, y, lx, ly = data
            x, y = x.to(self.device), y.to(self.device)

            with torch.inference_mode():
                h, attn = self.model(x, lx, y)
                loss = self.criterion(h.transpose(-1, -2), y)

            # two stats to compute (with tqdm bar)
            # one is loss
            total_loss += loss.item()
            # other is lv_distance
            vdist += calculate_levenshtein(h, y, lx, ly, self.decoder, phoneme_map)

            if i == 0: # print first three samples
                pred_strings = self.decoder.decode_prediction(h[:3,:,:])
                label_strings = [indices_to_chars(line, VOCAB) for line in y[:3,:].cpu().numpy()]
                plt.imshow(attn[0,:,:].cpu().detach().numpy())            
                plt.title('[epoch]%s\n%s\nGT:%s' % (epoch, pred_strings[0], label_strings[0]), fontdict = {'fontsize' : 12})
                plt.axis('off')
                plt.savefig('plots/' + self.run_id + '_epoch%d.png' % epoch)
                
            batch_bar.set_postfix(
                loss="{:.04f}".format(total_loss/ (i + 1)), 
                dist="{:.04f}".format(vdist / (i + 1))
            )
            batch_bar.update()

            del x, y, lx, ly, h, loss
            torch.cuda.empty_cache()

        batch_bar.close() #don't forget this!
        total_loss = total_loss/len(self.val_loader)
        val_dist = vdist/len(self.val_loader)
        return total_loss, val_dist
    
    def save_model(self, epoch, valid_loss, path):
        torch.save(
            {'model_state_dict'        : self.model.state_dict(),
            'optimizer_state_dict'     : self.optimizer.state_dict(),
            'scheduler_state_dict'     : self.lr_schedule.state_dict(),
            'valid_loss'               : valid_loss,
            'epoch'                    : epoch},
            path
        )
        