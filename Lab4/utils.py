from defines import LABELS, VOCAB, SOS_TOKEN, EOS_TOKEN
import Levenshtein
import ctcdecode
import torch
from abc import *

class DecodeUtilBase(metaclass=ABCMeta):
    @abstractmethod
    def decode_prediction(self, output, PHONEME_MAP=None):
        pass

# decoder using c++ ctcdecode library. (beam search)
class CTCDecodeUtil(DecodeUtilBase):
    def __init__(self, beam_width):
        super().__init__()
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(
            LABELS,
            model_path=None,
            alpha=0,
            beta=0,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=beam_width, #this should not exceed 50
            num_processes=4, #this should be ~#of cpus
            blank_id=0,
            log_probs_input=True
        )
        
    def decode_prediction(self, output, output_lens, PHONEME_MAP= LABELS):
        # decode output B*T*V(=VOCAB_SIZE) using beam search
        # CTC decoder
        beam_results, beam_scores, timesteps, output_lens  = self.ctc_decoder.decode(output, seq_lens= output_lens) #lengths - list of lengths
        # beam_scores -> small the better. returns in sorted order
        pred_strings  = []
        # output_lens: B * BW(beam width)
        for i in range(output_lens.shape[0]):
            str = ""
            for t in range(output_lens[i, 0].item()):
                # beam_results : B * BW(beam width) * T
                str += PHONEME_MAP[beam_results[i, 0, t].item()]
            pred_strings.append(str)
            
        return pred_strings

# decoder using greedy strategy
def indices_to_chars(indices, vocab):
    tokens = []
    for i in indices: # This loops through all the indices
        if int(i) == SOS_TOKEN: # If SOS is encountered, dont add it to the final list
            continue
        elif int(i) == EOS_TOKEN: # If EOS is encountered, stop the decoding process
            break
        else:
            tokens.append(vocab[i])
    return "".join(tokens)

class GreedyDecodeUtil(DecodeUtilBase):
    def decode_prediction(self, output, output_lens=None, PHONEME_MAP=VOCAB):
        pred_strings = []
        indices = torch.argmax(output, dim=-1)
        for i in range(indices.shape[0]):
            str = indices_to_chars(indices[i].tolist(), PHONEME_MAP)
            pred_strings.append(str)
        return pred_strings
    
#code from course notebook
def calculate_levenshtein(output, label, output_lens, label_lens, decoder, PHONEME_MAP= LABELS): # y - sequence of integers

    dist            = 0
    batch_size      = label.shape[0]

    pred_strings    = decoder.decode_prediction(output, output_lens, PHONEME_MAP)

    for i in range(batch_size):
        pred_string = pred_strings[i]
        label_string = ""
        for t in range(label_lens[i]):
            label_string += PHONEME_MAP[label[i, t].item()]
        dist += Levenshtein.distance(pred_string, label_string)

    dist /= batch_size
    return dist


# schedular classes
class CustomSchedular():
    def __init__(self, begin_value, begin_time, step_amount, step_every, value_limit=None) -> None:
        self.value = begin_value
        self.value_limit = value_limit
        self.prev = begin_time - step_every
        self.step_every = step_every
        self.step_amount = step_amount
        self.go = False
    
    def step(self, curr):
        if curr - self.prev >= self.step_every:
            self.prev = curr
            self.go = True
        #logic: step the value every call, in 5 mini-steps
        #assumes curr increments by 1
        if self.go and curr - self.prev < 5:
            ministep = self.step_amount/5
            if self.value_limit is not None and self.value - ministep < self.value_limit: return self.value
            self.value -= ministep
        elif self.go and curr - self.prev >= 5:
            #perform exact value calculation (remove fp-errors) here if needed
            self.go=False
        return self.value
    
    def get_value(self):
        return self.value
    
class LinearSchedular():
    def __init__(self, begin_value, begin_time, end_value, end_time):
        self.begin_value = begin_value
        self.end_value = end_value
        self.begin_time = begin_time
        self.end_time = end_time
        
        self.value = begin_value
        
    def step(self, t):
        if t > self.begin_time and t < self.end_time:
            self.value = self.begin_value + (t - self.begin_time)/(self.end_time - self.begin_time)*(self.end_value-self.begin_value)
        elif t >= self.end_time:
            self.value = self.end_value
            
        return self.get_value()
    
    def get_value(self):
        return self.value
    
    
# loading and saving script
def load_model(load_path, model, optimizer=None):

    print(f"Loading checkpoint from {load_path}")
    checkpoint  = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict= False)

    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.param_groups[0]['weight_decay'] = 1e-5

    epoch   = checkpoint['epoch']
    metric  = checkpoint['valid_loss']
    print(f"\tepoch{epoch} val_loss={metric:.04f}")

    return [model, optimizer, epoch, metric]
    